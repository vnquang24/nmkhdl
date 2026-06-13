#!/usr/bin/env python
"""Create report figures for the simplified RF + CNN results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, REPORTS_DIR


def _load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_model_comparison(results: dict, out_path: Path):
    labels, means, stds, colors = [], [], [], []
    spec = [
        ("cnn", "CNN 1D", "#4C78A8"),
        ("rf", "Random Forest", "#59A14F"),
        ("svm", "SVM (RBF)", "#B279A2"),
    ]
    for key, label, color in spec:
        if key in results and "mean_macroF1" in results[key]:
            labels.append(label)
            means.append(results[key]["mean_macroF1"])
            stds.append(results[key].get("std_macroF1", 0.0))
            colors.append(color)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, width=0.55)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("So sánh macro-F1 ba mô hình (3-fold)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fold_scores(results: dict, out_path: Path):
    spec = [
        ("cnn", "CNN 1D", "#4C78A8"),
        ("rf", "Random Forest", "#59A14F"),
        ("svm", "SVM (RBF)", "#B279A2"),
    ]
    series = [(label, results[key].get("fold_f1s", []), color)
              for key, label, color in spec
              if key in results and results[key].get("fold_f1s")]
    if not series:
        return
    n_folds = max(len(f) for _, f, _ in series)
    folds = np.arange(n_folds)
    n = len(series)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (label, f1s, color) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        ax.bar(folds + offset, f1s, width, label=label, color=color)

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i + 1}" for i in folds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Độ ổn định macro-F1 qua 3 fold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rf_accuracy(results: dict, out_path: Path):
    accs = results["rf"].get("fold_accuracies", [])
    mean_acc = results["rf"].get("mean_accuracy")
    if not accs:
        return

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(accs))
    ax.plot(x, accs, marker="o", linewidth=2.5, color="#E15759", label="Accuracy từng fold")
    ax.axhline(mean_acc, linestyle="--", color="#333333", linewidth=1.5, label=f"Trung bình {mean_acc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in x])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest accuracy qua các fold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    result_path = REPORTS_DIR / "simple_results.json"
    results = _load_results(result_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "comparison": FIGURES_DIR / "simple_model_comparison.png",
        "folds": FIGURES_DIR / "simple_fold_macro_f1.png",
        "rf_accuracy": FIGURES_DIR / "simple_rf_accuracy.png",
    }
    plot_model_comparison(results, outputs["comparison"])
    plot_fold_scores(results, outputs["folds"])
    plot_rf_accuracy(results, outputs["rf_accuracy"])

    print("Saved figures:")
    for path in outputs.values():
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
