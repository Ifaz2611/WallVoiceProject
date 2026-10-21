"""Plot training and evaluation metrics, hyperparameter curves, and loss/accuracy charts.

Generates publication-quality visual charts for the WallVoice24BD pipeline:
- outputs/training-performance.png:
    * Panel A: Model comparison (Validation Accuracy, Macro-F1, Classification Error / 0-1 Loss)
    * Panel B: Hyperparameter tuning curve (CV Macro-F1 and Accuracy vs k-neighbors)
    * Panel C: Per-class validation performance (Precision, Recall, F1 for each area)
    * Panel D: Validation confusion matrix (percentages and counts)
- outputs/evaluation-performance.png (when test evaluation exists):
    * Panel A: Test per-class performance (Precision, Recall, F1)
    * Panel B: Test normalized confusion matrix heatmap

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import common

# Use a clean, publication-ready style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "figure.autolayout": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-selection",
        default=str(common.OUTPUTS_DIR / "model-selection.json"),
        help="Path to model-selection.json",
    )
    parser.add_argument(
        "--evaluation",
        default=str(common.OUTPUTS_DIR / "evaluation.json"),
        help="Path to evaluation.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(common.OUTPUTS_DIR),
        help="Directory to save generated chart images",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive plot window (requires GUI display).",
    )
    return parser.parse_args()


def plot_training_dashboard(
    model_selection_path: Path,
    output_dir: Path,
    show: bool = False,
) -> Path | None:
    """Plot multi-panel training and model selection performance dashboard."""
    if not model_selection_path.exists():
        print(f"Model selection data not found at: {model_selection_path}")
        return None

    with model_selection_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # ------------------------------------------------------------------
    # Panel 1: Model & Feature Extractor Comparison (Acc, Macro-F1, Error/Loss)
    # ------------------------------------------------------------------
    models_labels = []
    accuracies = []
    macro_f1s = []
    error_rates = []

    # KNN Extractors
    for res in data.get("results", []):
        ext = res["extractor"].upper()
        models_labels.append(f"KNN ({ext})")
        acc = res["val_metrics"]["accuracy"]
        f1 = res["val_metrics"]["macro_f1"]
        accuracies.append(acc)
        macro_f1s.append(f1)
        error_rates.append(1.0 - acc)

    # Baselines
    baseline_display_names = {
        "logistic_regression_balanced": "Logistic Reg (bal)",
        "linear_svm_balanced": "Linear SVM (bal)",
        "dummy_most_frequent": "Dummy (Frequent)",
    }
    for b_key, b_metrics in data.get("baselines", {}).items():
        name = baseline_display_names.get(b_key, b_key)
        models_labels.append(name)
        acc = b_metrics["accuracy"]
        f1 = b_metrics["macro_f1"]
        accuracies.append(acc)
        macro_f1s.append(f1)
        error_rates.append(1.0 - acc)

    x = np.arange(len(models_labels))
    width = 0.26

    ax1 = axes[0]
    rects1 = ax1.bar(x - width, accuracies, width, label="Accuracy", color="#2b5c8f")
    rects2 = ax1.bar(x, macro_f1s, width, label="Macro-F1", color="#2a9d8f")
    rects3 = ax1.bar(x + width, error_rates, width, label="0-1 Error Rate (Loss)", color="#e76f51")

    ax1.set_title("A. Model Comparison (Validation Accuracy, Macro-F1, Error Rate)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_labels, rotation=18, ha="right")
    ax1.set_ylabel("Metric Score (0.0 - 1.0)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Value labels
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # ------------------------------------------------------------------
    # Panel 2: Hyperparameter Tuning Curves (CV Macro-F1 vs k across PCA)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    best_extractor = data.get("best_extractor", "hog")
    cv_csv_path = common.OUTPUTS_DIR / f"cv-results-{best_extractor}.csv"

    if cv_csv_path.exists():
        cv_df = pd.read_csv(cv_csv_path)
        pca_values = sorted(cv_df["param_pca__n_components"].unique())
        palette = sns.color_palette("deep", len(pca_values))

        for idx, pca_comp in enumerate(pca_values):
            subset = cv_df[cv_df["param_pca__n_components"] == pca_comp].copy()
            # Aggregate over weights / metric by taking best for each k
            k_agg = (
                subset.groupby("param_knn__n_neighbors")
                .agg(
                    mean_f1=("mean_test_macro_f1", "max"),
                    mean_acc=("mean_test_accuracy", "max"),
                )
                .reset_index()
                .sort_values("param_knn__n_neighbors")
            )

            ax2.plot(
                k_agg["param_knn__n_neighbors"],
                k_agg["mean_f1"],
                marker="o",
                linewidth=2,
                color=palette[idx],
                label=f"PCA {pca_comp} (Macro-F1)",
            )

        ax2.set_title(f"B. Hyperparameter Tuning Curves for {best_extractor.upper()} (CV Macro-F1 vs k)")
        ax2.set_xlabel("Number of Neighbors (k)")
        ax2.set_ylabel("CV Macro-F1")
        ax2.set_xticks(sorted(cv_df["param_knn__n_neighbors"].unique()))
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.7)
    else:
        ax2.text(0.5, 0.5, f"No CV results file found for {best_extractor}", ha="center", va="center")
        ax2.set_title("B. Hyperparameter Tuning Curves")

    # ------------------------------------------------------------------
    # Panel 3: Per-Class Performance Breakdown (Validation)
    # ------------------------------------------------------------------
    ax3 = axes[2]
    val_per_class = data.get("validation_per_class", {})
    class_labels = sorted(val_per_class.keys())

    if class_labels:
        precisions = [val_per_class[k]["precision"] for k in class_labels]
        recalls = [val_per_class[k]["recall"] for k in class_labels]
        f1_scores = [val_per_class[k]["f1"] for k in class_labels]
        supports = [val_per_class[k]["support"] for k in class_labels]

        x_cls = np.arange(len(class_labels))
        w = 0.25

        rects_p = ax3.bar(x_cls - w, precisions, w, label="Precision", color="#457b9d")
        rects_r = ax3.bar(x_cls, recalls, w, label="Recall", color="#e9c46a")
        rects_f = ax3.bar(x_cls + w, f1_scores, w, label="F1-Score", color="#2a9d8f")

        ax3.set_title(f"C. Per-Class Validation Performance ({best_extractor.upper()})")
        ax3.set_xticks(x_cls)
        xtick_labels = [f"{lbl}\n(n={supports[i]})" for i, lbl in enumerate(class_labels)]
        ax3.set_xticklabels(xtick_labels)
        ax3.set_xlabel("Collection Area (Support Count in Val)")
        ax3.set_ylabel("Score (0.0 - 1.0)")
        ax3.set_ylim(0, 1.1)
        ax3.legend(loc="upper right")
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        for rects in [rects_p, rects_r, rects_f]:
            for rect in rects:
                h = rect.get_height()
                ax3.annotate(
                    f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    else:
        ax3.text(0.5, 0.5, "No per-class metrics found", ha="center", va="center")
        ax3.set_title("C. Per-Class Performance")

    # ------------------------------------------------------------------
    # Panel 4: Validation Confusion Matrix (Normalized Percentages)
    # ------------------------------------------------------------------
    ax4 = axes[3]
    best_res = next((r for r in data.get("results", []) if r["extractor"] == best_extractor), None)
    if best_res and "confusion_matrix" in best_res["val_metrics"]:
        matrix = np.array(best_res["val_metrics"]["confusion_matrix"])
        labels = data.get("labels", class_labels)

        # Row-normalized matrix (recall per cell)
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(
            matrix.astype(float),
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        ) * 100.0

        im = ax4.imshow(norm_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label("Row Normalized % (Recall)")

        ax4.set_xticks(np.arange(len(labels)))
        ax4.set_yticks(np.arange(len(labels)))
        ax4.set_xticklabels(labels)
        ax4.set_yticklabels(labels)
        ax4.set_xlabel("Predicted Area")
        ax4.set_ylabel("True Area")
        ax4.set_title(f"D. Validation Confusion Matrix ({best_extractor.upper()})")

        threshold = 50.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                raw_val = matrix[i, j]
                pct_val = norm_matrix[i, j]
                text_color = "white" if pct_val > threshold else "black"
                ax4.text(
                    j,
                    i,
                    f"{pct_val:.1f}%\n({raw_val})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
    else:
        ax4.text(0.5, 0.5, "No confusion matrix found", ha="center", va="center")
        ax4.set_title("D. Validation Confusion Matrix")

    fig.suptitle(
        "WallVoice24BD Model Training & Validation Dashboard",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = output_dir / "training-performance.png"
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_evaluation_dashboard(
    evaluation_path: Path,
    output_dir: Path,
    show: bool = False,
) -> Path | None:
    """Plot test set evaluation metrics and normalized confusion matrix."""
    if not evaluation_path.exists():
        return None

    with evaluation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    labels = data.get("labels", [])
    per_class = metrics.get("per_class", {})
    matrix = np.array(metrics.get("confusion_matrix", []))

    acc = metrics.get("accuracy", 0.0)
    macro_f1 = metrics.get("macro_f1", 0.0)
    weighted_f1 = metrics.get("weighted_f1", 0.0)
    error_rate = 1.0 - acc
    n_test = data.get("n_test", 0)
    misclassified = data.get("misclassified", 0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ------------------------------------------------------------------
    # Panel 1: Per-Class Test Performance (Precision, Recall, F1)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    class_labels = sorted(per_class.keys()) if per_class else labels
    precisions = [per_class[k]["precision"] for k in class_labels]
    recalls = [per_class[k]["recall"] for k in class_labels]
    f1_scores = [per_class[k]["f1"] for k in class_labels]
    supports = [per_class[k]["support"] for k in class_labels]

    x = np.arange(len(class_labels))
    w = 0.26
    r1 = ax1.bar(x - w, precisions, w, label="Precision", color="#457b9d")
    r2 = ax1.bar(x, recalls, w, label="Recall", color="#e9c46a")
    r3 = ax1.bar(x + w, f1_scores, w, label="F1-Score", color="#2a9d8f")

    ax1.set_title("A. Test Per-Class Performance (Precision / Recall / F1)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lbl}\n(n={supports[i]})" for i, lbl in enumerate(class_labels)])
    ax1.set_xlabel("Area (Test Support)")
    ax1.set_ylabel("Score (0.0 - 1.0)")
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for rects in [r1, r2, r3]:
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # ------------------------------------------------------------------
    # Panel 2: Normalized Confusion Matrix Heatmap (Test)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    if matrix.size > 0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(
            matrix.astype(float),
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        ) * 100.0

        im = ax2.imshow(norm_matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Row Normalized % (Recall)")

        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_yticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel("Predicted Area")
        ax2.set_ylabel("True Area")
        ax2.set_title("B. Test Confusion Matrix (Percentages & Counts)")

        threshold = 50.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                raw_val = matrix[i, j]
                pct_val = norm_matrix[i, j]
                text_color = "white" if pct_val > threshold else "black"
                ax2.text(
                    j,
                    i,
                    f"{pct_val:.1f}%\n({raw_val})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    subtitle = (
        f"Test Accuracy: {acc:.4f} ({acc*100:.1f}%) | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Weighted-F1: {weighted_f1:.4f} | "
        f"0-1 Error: {error_rate:.4f} ({misclassified}/{n_test} errors)"
    )
    fig.suptitle(
        f"WallVoice24BD Frozen Test Set Evaluation\n{subtitle}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = output_dir / "evaluation-performance.png"
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    model_sel_path = Path(args.model_selection)
    eval_path = Path(args.evaluation)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating performance plots...")
    p1 = plot_training_dashboard(model_sel_path, out_dir, show=args.show)
    if p1:
        print(f"  Training plot saved: {p1.relative_to(common.ROOT)}")

    p2 = plot_evaluation_dashboard(eval_path, out_dir, show=args.show)
    if p2:
        print(f"  Evaluation plot saved: {p2.relative_to(common.ROOT)}")
    else:
        print("  (Note: evaluation.json not found or not yet generated; skipping test evaluation plot)")

    print("Plotting complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
