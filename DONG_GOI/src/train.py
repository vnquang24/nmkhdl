"""Config-driven training pipeline cho thí nghiệm thử-và-sai.

Tham số chính của `cfg` (dict):
    name, phase, notes
    n_splits=3, epochs=40, batch_size=128, num_workers=0, seed=42
    optim:    {type: adam|adamw, lr, weight_decay, betas}
    sched:    {type: none|cosine|onecycle, t_max, max_lr}
    loss:     {type: ce|focal|labelsmooth, gamma, smoothing,
               class_weight: none|invfreq|sqrtinvfreq|effnum, beta}
    augment:  {enabled, jitter_sigma, scale_low/high, rot_theta,
               use_rotation, use_jitter, use_scale, use_time_warp, ...,
               mixup_alpha, cutmix_alpha}      # mixup/cutmix=0 → tắt
    model:    {type, ...}                       # truyền thẳng cho models.make_model
    early_stopping_patience: 0 (tắt) hoặc int

Hàm chính: `run_cv(cfg)` — chạy 3-fold + log qua ExperimentLogger.
"""
from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from .config import PROCESSED_DIR, MODELS_DIR, SEED
from .datasets import InertialDataset, compute_scaler, mixup_batch, cutmix_batch
from .models import make_model
from .experiment import ExperimentLogger


# --------------------------- I/O ---------------------------

def load_npz(path: Path | None = None) -> dict:
    path = path or (PROCESSED_DIR / "windows.npz")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# --------------------------- Loss & weights ---------------------------

def _class_weights(y: np.ndarray, n_classes: int, strategy: str,
                   beta: float = 0.999) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if strategy == "none":
        return np.ones(n_classes, dtype=np.float32)
    if strategy == "invfreq":
        w = counts.sum() / counts
    elif strategy == "sqrtinvfreq":
        w = np.sqrt(counts.sum() / counts)
    elif strategy == "effnum":
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-9)
    else:
        raise ValueError(f"unknown class_weight strategy {strategy}")
    w = w / w.mean()
    return w.astype(np.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        nll = F.nll_loss(((1 - p) ** self.gamma) * logp, target,
                         weight=self.weight, reduction="mean")
        return nll


def _build_loss(cfg: dict, weights: torch.Tensor) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    t = loss_cfg.get("type", "ce").lower()
    if t == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    if t == "labelsmooth":
        return nn.CrossEntropyLoss(weight=weights,
                                   label_smoothing=loss_cfg.get("smoothing", 0.1))
    if t == "focal":
        return FocalLoss(weight=weights, gamma=loss_cfg.get("gamma", 2.0))
    raise ValueError(f"unknown loss {t}")


def _build_optimizer(model, cfg):
    o = cfg.get("optim", {})
    lr = o.get("lr", 1e-3)
    wd = o.get("weight_decay", 1e-4)
    t = o.get("type", "adam").lower()
    if t == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if t == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optim {t}")


def _build_scheduler(optim, cfg, steps_per_epoch: int, epochs: int):
    s = cfg.get("sched", {})
    t = s.get("type", "none").lower()
    if t == "none":
        return None
    if t == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=s.get("t_max", epochs))
    if t == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=s.get("max_lr", 3e-3),
            steps_per_epoch=steps_per_epoch, epochs=epochs)
    raise ValueError(f"unknown sched {t}")


# --------------------------- Training loop ---------------------------

def _train_one_fold(X_tr, y_tr, X_va, y_va, n_classes, device, cfg, verbose=True):
    mean, std = compute_scaler(X_tr)
    aug_cfg = cfg.get("augment", {}) or {}
    aug_enabled = aug_cfg.get("enabled", True)
    mixup_alpha = aug_cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = aug_cfg.get("cutmix_alpha", 0.0)

    ds_tr = InertialDataset(X_tr, y_tr, mean, std,
                            augment=aug_enabled, aug_cfg=aug_cfg, rng_seed=cfg.get("seed", SEED))
    ds_va = InertialDataset(X_va, y_va, mean, std, augment=False)

    bs = cfg.get("batch_size", 128)
    nw = cfg.get("num_workers", 0)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw,
                       pin_memory=(device.type == "cuda"), drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw,
                       pin_memory=(device.type == "cuda"))

    in_ch = X_tr.shape[-1]
    model = make_model(cfg.get("model", {"type": "cnn"}), in_ch, n_classes).to(device)

    cw_strategy = cfg.get("loss", {}).get("class_weight", "invfreq")
    cw = _class_weights(y_tr, n_classes, cw_strategy,
                        beta=cfg.get("loss", {}).get("beta", 0.999))
    cw_t = torch.from_numpy(cw).to(device)
    criterion = _build_loss(cfg, cw_t)
    optim = _build_optimizer(model, cfg)
    epochs = cfg.get("epochs", 40)
    sched = _build_scheduler(optim, cfg, steps_per_epoch=max(1, len(dl_tr)),
                             epochs=epochs)

    use_amp = (device.type == "cuda") and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.default_rng(cfg.get("seed", SEED))
    best_f1, best_state = -1.0, None
    patience = cfg.get("early_stopping_patience", 0)
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, n = 0.0, 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()

            # mixup / cutmix (only one active per step)
            use_mix = mixup_alpha > 0 and rng.random() < 0.5
            use_cut = (not use_mix) and cutmix_alpha > 0 and rng.random() < 0.5

            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_mix:
                    xb_m, ya, yb2, lam = mixup_batch(xb, yb, mixup_alpha, rng)
                    out = model(xb_m)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                elif use_cut:
                    xb_m, ya, yb2, lam = cutmix_batch(xb, yb, cutmix_alpha, rng)
                    out = model(xb_m)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
                else:
                    out = model(xb)
                    loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            if sched is not None and cfg.get("sched", {}).get("type") == "onecycle":
                sched.step()

            tot_loss += float(loss) * xb.size(0)
            n += xb.size(0)
        if sched is not None and cfg.get("sched", {}).get("type") != "onecycle":
            sched.step()

        # eval
        model.eval()
        all_p, all_t = [], []
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(xb)
                    loss = criterion(out, yb)
                val_loss += float(loss) * xb.size(0)
                nv += xb.size(0)
                all_p.append(out.argmax(dim=-1).cpu().numpy())
                all_t.append(yb.cpu().numpy())
        all_p = np.concatenate(all_p)
        all_t = np.concatenate(all_t)
        f1 = float(f1_score(all_t, all_p, average="macro", zero_division=0))

        rec = {"epoch": epoch,
               "train_loss": tot_loss / max(1, n),
               "val_loss": val_loss / max(1, nv),
               "val_macroF1": f1,
               "lr": optim.param_groups[0]["lr"]}
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch:3d}  trL={rec['train_loss']:.4f}  "
                  f"vaL={rec['val_loss']:.4f}  F1={f1:.4f}  lr={rec['lr']:.2e}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (no improve {patience} epochs)")
                break

    return best_f1, best_state, history, mean, std


def run_cv(cfg: dict, verbose: bool = True) -> dict:
    """Chạy StratifiedGroupKFold CV, log qua ExperimentLogger, trả về dict tổng kết."""
    cfg = dict(cfg)  # shallow copy
    name = cfg.get("name", "unnamed")
    phase = cfg.get("phase", "X")
    notes = cfg.get("notes", "")
    seed = cfg.get("seed", SEED)
    n_splits = cfg.get("n_splits", 3)

    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_npz()
    X, y, activity, groups = (
        data["X"], data["y"], data["activity"], data["groups"])
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    if verbose:
        print(f"[run_cv] {name} | windows={len(X)} classes={n_classes} folds={n_splits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[run_cv] device={device}")

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_f1s = []
    with ExperimentLogger(name=name, phase=phase, config=cfg, notes=notes) as run:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_idx, groups=groups)):
            if verbose:
                print(f"[fold {fold}] train={len(tr_idx)}  val={len(va_idx)}")
            X_tr, y_tr = X[tr_idx], y_idx[tr_idx]
            X_va, y_va = X[va_idx], y_idx[va_idx]
            t0 = time.time()
            best_f1, state, hist, mean, std = _train_one_fold(
                X_tr, y_tr, X_va, y_va, n_classes, device, cfg, verbose=verbose)
            dt = time.time() - t0
            fold_f1s.append(best_f1)
            run.set_metric(f"fold{fold}_f1", best_f1)
            run.set_metric(f"fold{fold}_runtime", round(dt, 1))
            # save fold artifacts under run dir
            fold_dir = run.run_dir / f"fold{fold}"
            fold_dir.mkdir(exist_ok=True)
            torch.save({"state_dict": state,
                        "mean": mean, "std": std,
                        "classes": classes}, fold_dir / "model.pt")
            with open(fold_dir / "history.json", "w") as f:
                json.dump(hist, f, indent=2)
            if verbose:
                print(f"[fold {fold}] best_F1={best_f1:.4f}  runtime={dt:.1f}s")
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        run.set_metric("mean_macroF1", mean_f1)
        run.set_metric("std_macroF1", std_f1)
        if verbose:
            print(f"[run_cv] {name}  mean_macroF1={mean_f1:.4f} ± {std_f1:.4f}")
        return {"name": name, "fold_f1s": fold_f1s,
                "mean_macroF1": mean_f1, "std_macroF1": std_f1,
                "run_id": run.run_id}


# --------------------------- CLI ---------------------------

def _main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_cv(cfg)


if __name__ == "__main__":
    _main()
