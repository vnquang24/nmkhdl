#!/usr/bin/env python
"""Sinh các figure minh hoạ bổ sung cho báo cáo:

1. preprocess_raw_vs_filtered.png  - tín hiệu thô vs sau lọc Butterworth.
2. window_segmentation.png         - minh hoạ sliding window 128/overlap 50%.
3. split_scheme.png                - sơ đồ StratifiedGroupKFold theo group user__file.
4. rf_feature_importance.png       - top-20 đặc trưng quan trọng của Random Forest.
5. feature_pca.png                 - chiếu 65 đặc trưng xuống 2D bằng PCA.
6. rf_confusion.png                - ma trận nhầm lẫn 15x15 của Random Forest.

Chạy: venv/Scripts/python.exe -m scripts.plot_report_extra
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (FIGURES_DIR, FS, LOWPASS_CUTOFF, HIGHPASS_GRAVITY,
                        WINDOW_SIZE, WINDOW_STRIDE, INERTIAL_CHANNELS, SEED)
from src.io import load_inertial
from src.preprocess import preprocess_inertial, _butter_lowpass
from src.features import batch_features
from src.train import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix


# --------------------------------------------------------------------------
def plot_preprocess(out_path: Path):
    """Tín hiệu acc_x thô vs sau lọc, trên một đoạn walking của userA."""
    df = load_inertial("userA", "inertial_walking_s1_att1.csv")
    raw = df[INERTIAL_CHANNELS].to_numpy(dtype=np.float32)
    # Đoạn 4 giây để nhìn rõ dao động
    n = int(4 * FS)
    seg_raw = raw[:n]
    # Lọc low-pass cho toàn chuỗi rồi cắt cùng đoạn (tránh hiệu ứng biên)
    filtered = preprocess_inertial(df, fs_in=FS)
    seg_fil = filtered[:n]
    t = np.arange(len(seg_raw)) / FS

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    # acc_x: thô vs low-pass (đã loại trọng lực qua high-pass nên khác mức DC)
    axes[0].plot(t, seg_raw[:, 0], color="#BBBBBB", lw=1.0, label="acc\\_x thô")
    axes[0].plot(t, seg_fil[:, 0], color="#4C78A8", lw=1.4, label="acc\\_x sau lọc")
    axes[0].set_ylabel("Gia tốc (m/s$^2$)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Gia tốc trục x: trước và sau tiền xử lý (userA, walking)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, seg_raw[:, 3], color="#BBBBBB", lw=1.0, label="gyro\\_x thô")
    axes[1].plot(t, seg_fil[:, 3], color="#E15759", lw=1.4, label="gyro\\_x sau lọc")
    axes[1].set_ylabel("Vận tốc góc (rad/s)")
    axes[1].set_xlabel("Thời gian (giây)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_segmentation(out_path: Path):
    """Minh hoạ cửa sổ trượt 128 mẫu, stride 64 (overlap 50%)."""
    df = load_inertial("userA", "inertial_walking_s1_att1.csv")
    sig = preprocess_inertial(df, fs_in=FS)[:, 0]
    n = WINDOW_SIZE * 3  # đủ cho ~5 cửa sổ
    sig = sig[:n]
    t = np.arange(len(sig)) / FS

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, sig, color="#333333", lw=1.0)
    colors = ["#4C78A8", "#59A14F", "#E15759", "#B279A2", "#F28E2B"]
    n_win = 1 + (len(sig) - WINDOW_SIZE) // WINDOW_STRIDE
    for i in range(min(n_win, 5)):
        s = i * WINDOW_STRIDE
        e = s + WINDOW_SIZE
        ax.axvspan(s / FS, e / FS, ymin=0.02 + 0.16 * i, ymax=0.16 + 0.16 * i,
                   color=colors[i % len(colors)], alpha=0.35)
        ax.text((s + WINDOW_SIZE / 2) / FS, ax.get_ylim()[1] * (0.92 - 0.0 * i),
                f"W{i+1}", color=colors[i % len(colors)], fontsize=9,
                ha="center", fontweight="bold")
    ax.set_xlabel("Thời gian (giây)")
    ax.set_ylabel("acc\\_x sau lọc")
    ax.set_title(f"Cửa sổ trượt {WINDOW_SIZE} mẫu, bước {WINDOW_STRIDE} (overlap 50\\%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_split_scheme(out_path: Path):
    """Sơ đồ minh hoạ StratifiedGroupKFold: cả cửa sổ của 1 file nằm trọn 1 phía."""
    rng = np.random.default_rng(SEED)
    n_files = 12
    fold_of_file = rng.integers(0, 3, size=n_files)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fold_colors = {0: "#4C78A8", 1: "#59A14F", 2: "#E15759"}
    x = 0
    for f in range(n_files):
        n_win = rng.integers(3, 8)  # số cửa sổ của file
        for w in range(n_win):
            ax.add_patch(mpatches.Rectangle((x, 0), 0.9, 1.0,
                         color=fold_colors[fold_of_file[f]], alpha=0.85))
            x += 1
        # vạch ngăn giữa các file
        ax.axvline(x - 0.05, color="white", lw=1.5)
        ax.text(x - n_win / 2 - 0.5, -0.45, f"file{f+1}", ha="center",
                fontsize=7, rotation=0)
        x += 0.4
    ax.set_xlim(-0.5, x)
    ax.set_ylim(-0.9, 1.5)
    ax.axis("off")
    ax.set_title("StratifiedGroupKFold: mọi cửa sổ của cùng một file thuộc cùng một fold")
    handles = [mpatches.Patch(color=fold_colors[i], label=f"Fold {i+1}") for i in range(3)]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_features():
    data = load_npz()
    X, y, groups = data["X"], data["y"], data["groups"]
    F, names = batch_features(X)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    classes, y_idx = np.unique(y, return_inverse=True)
    return F, names, y, y_idx, classes, groups


def plot_rf_importance(F, names, y_idx, out_path: Path, top=20):
    sc = StandardScaler().fit(F)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                 class_weight="balanced_subsample")
    clf.fit(sc.transform(F), y_idx)
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 6))
    yp = np.arange(len(order))[::-1]
    ax.barh(yp, imp[order], color="#59A14F")
    ax.set_yticks(yp)
    ax.set_yticklabels([names[i].replace("_", "\\_") for i in order], fontsize=8)
    ax.set_xlabel("Mức độ quan trọng (Gini importance)")
    ax.set_title(f"Top-{top} đặc trưng quan trọng nhất của Random Forest")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_pca(F, y, classes, out_path: Path):
    sc = StandardScaler().fit(F)
    Z = PCA(n_components=2, random_state=SEED).fit_transform(sc.transform(F))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cmap = plt.get_cmap("tab20", len(classes))
    for k, c in enumerate(classes):
        m = y == c
        # giảm mật độ điểm để hình nhẹ
        idx = np.where(m)[0]
        if len(idx) > 400:
            idx = idx[:: max(1, len(idx) // 400)]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=6, color=cmap(k), label=c, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Chiếu 65 đặc trưng xuống 2D bằng PCA (mỗi màu = một người dùng)")
    ax.legend(ncol=3, fontsize=7, markerscale=2, loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rf_confusion(F, y_idx, groups, classes, out_path: Path, n_splits=3):
    """Gom prediction RF qua 3 fold (out-of-fold) rồi vẽ confusion 15x15."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    gts, preds = [], []
    for tr, va in skf.split(F, y_idx, groups=groups):
        sc = StandardScaler().fit(F[tr])
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                     class_weight="balanced_subsample")
        clf.fit(sc.transform(F[tr]), y_idx[tr])
        preds.append(clf.predict(sc.transform(F[va])))
        gts.append(y_idx[va])
    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    cm = confusion_matrix(gts, preds, labels=list(range(len(classes))))
    cmn = cm.astype(np.float32) / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title("Ma trận nhầm lẫn Random Forest (chuẩn hoá theo hàng)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("1/6 preprocess ...")
    plot_preprocess(FIGURES_DIR / "preprocess_raw_vs_filtered.png")
    print("2/6 window segmentation ...")
    plot_window_segmentation(FIGURES_DIR / "window_segmentation.png")
    print("3/6 split scheme ...")
    plot_split_scheme(FIGURES_DIR / "split_scheme.png")

    print("Loading features for model figures ...")
    F, names, y, y_idx, classes, groups = _load_features()
    print("4/6 RF feature importance ...")
    plot_rf_importance(F, names, y_idx, FIGURES_DIR / "rf_feature_importance.png")
    print("5/6 feature PCA ...")
    plot_feature_pca(F, y, classes, FIGURES_DIR / "feature_pca.png")
    print("6/6 RF confusion ...")
    plot_rf_confusion(F, y_idx, groups, classes, FIGURES_DIR / "rf_confusion.png")

    print("Done. Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
