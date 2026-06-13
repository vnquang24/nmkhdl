#!/usr/bin/env python
"""Run the simplified CNN + RF + SVM experiment for the project report.

This script intentionally avoids the old sweep-style configs. It trains one
plain CNN 1D configuration, one Random Forest baseline and one SVM (RBF)
baseline on the full closed-set of 15 users, then writes a small JSON summary
under reports/simple_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR
from src.evaluate import rf_baseline, svm_baseline
from src.train import load_npz, run_cv


DEFAULT_CONFIG = Path("experiments/configs/SIMPLE_CNN.yaml")


def _load_labels():
    data = load_npz()
    X, y, _activity, groups = (
        data["X"],
        data["y"],
        data["activity"],
        data["groups"],
    )
    classes = sorted(np.unique(y).tolist())
    cls2id = {c: i for i, c in enumerate(classes)}
    y_id = np.array([cls2id[v] for v in y])
    return X, y_id, groups, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_path = REPORTS_DIR / "simple_results.json"
    if out_path.exists() and (args.skip_cnn or args.skip_rf or args.skip_svm):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    results.update({
        "config": str(cfg_path),
        "method": "Simple baseline: CNN 1D + Random Forest + SVM (RBF)",
    })

    if not args.skip_cnn:
        print("=== Train simple CNN 1D ===")
        results["cnn"] = run_cv(cfg, verbose=not args.quiet)

    if not (args.skip_rf and args.skip_svm):
        X, y_id, groups, classes = _load_labels()
        results["classes"] = classes
        n_splits = cfg.get("n_splits", 3)

        if not args.skip_rf:
            print("\n=== Train Random Forest baseline ===")
            results["rf"] = rf_baseline(X, y_id, groups, n_splits=n_splits)
            print(
                "RF mean_macroF1="
                f"{results['rf']['mean_macroF1']:.4f} ± {results['rf']['std_macroF1']:.4f}"
            )

        if not args.skip_svm:
            print("\n=== Train SVM (RBF) baseline ===")
            results["svm"] = svm_baseline(X, y_id, groups, n_splits=n_splits)
            print(
                "SVM mean_macroF1="
                f"{results['svm']['mean_macroF1']:.4f} ± {results['svm']['std_macroF1']:.4f}"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
