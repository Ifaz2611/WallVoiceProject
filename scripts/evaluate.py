"""Evaluate the frozen KNN model once on the test split.

Refuses to overwrite a previous test evaluation unless ``--force`` is passed, so
the reported test numbers stay honest. Writes metrics, a classification report,
per-image predictions, a confusion matrix, and a contact sheet of mistakes.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/knn_best.joblib
"""

from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(common.MODELS_DIR / "knn_best.joblib"))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing test evaluation.")
    parser.add_argument("--max-mistakes", type=int, default=24)
    return parser.parse_args()


def load_test(extractor: str):
    feature_path = common.FEATURES_DIR / f"{extractor}_test.npy"
    ids_path = common.FEATURES_DIR / f"{extractor}_test_ids.csv"
    if not feature_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Missing test features for {extractor}. Run: python scripts/extract_features.py --extractor {extractor}"
        )
    features = np.load(feature_path)
    ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
    if len(features) != len(ids):
        raise ValueError("Test feature/ID row mismatch")
    return features, ids


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="KNN confusion matrix (test split)",
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_mistakes(ids: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray | None, path, limit: int) -> int:
    mistaken = np.flatnonzero(ids["label"].to_numpy() != predictions)
    if len(mistaken) == 0:
        return 0
    chosen = mistaken[:limit]
    columns = 6
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis in axes:
        axis.axis("off")
    for axis, index in zip(axes, chosen):
        record = ids.iloc[index]
        image = common.open_rgb(common.image_path(record.relative_path))
        axis.imshow(image)
        title = f"true {record.label} -> pred {predictions[index]}"
        if probabilities is not None:
            title += f"\nconfidence {probabilities[index]:.2f}"
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    figure.suptitle("Misclassified test images", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return len(chosen)


def main() -> int:
    args = parse_args()
    output_json = common.OUTPUTS_DIR / "evaluation.json"
    if output_json.exists() and not args.force:
        raise SystemExit(
            "outputs/evaluation.json already exists; the test split has already been evaluated. "
            "Use --force only to reproduce the same frozen-split run."
        )

    bundle = joblib.load(args.model)
    extractor = bundle["extractor"]
    labels = list(bundle.get("labels") or sorted(bundle["pipeline"].classes_))
    features, ids = load_test(extractor)

    model = bundle["pipeline"]
    predictions = model.predict(features)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features).max(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        ids["label"], predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(ids["label"], predictions, labels=labels)
    report_text = classification_report(ids["label"], predictions, labels=labels, zero_division=0, digits=3)

    evaluation = {
        "model": str(common.ROOT / args.model),
        "extractor": extractor,
        "best_params": bundle.get("best_params"),
        "labels": labels,
        "n_test": int(len(ids)),
        "test_label_counts": common.label_counts(ids["label"]),
        "metrics": {
            "accuracy": float(accuracy_score(ids["label"], predictions)),
            "macro_f1": float(f1_score(ids["label"], predictions, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(ids["label"], predictions, labels=labels, average="weighted", zero_division=0)
            ),
            "per_class": {
                label: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(labels)
            },
            "confusion_matrix": matrix.tolist(),
        },
        "split": "splits/test.csv (frozen, evaluated once)",
        "versions": common.environment_versions(),
    }
    common.write_json(output_json, evaluation)
    (common.OUTPUTS_DIR / "classification-report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix(matrix, labels, common.OUTPUTS_DIR / "confusion-matrix-test.png")

    prediction_table = ids[["image_id", "relative_path", "label", "group_id"]].copy()
    prediction_table["prediction"] = predictions
    prediction_table["correct"] = prediction_table["label"] == prediction_table["prediction"]
    if probabilities is not None:
        prediction_table["confidence"] = np.round(probabilities, 4)
    prediction_table.to_csv(common.OUTPUTS_DIR / "test-predictions.csv", index=False)

    mistakes_saved = save_mistakes(
        ids, predictions, probabilities, common.OUTPUTS_DIR / "misclassified-test.png", args.max_mistakes
    )
    evaluation["misclassified"] = int((~prediction_table["correct"]).sum())
    evaluation["mistakes_in_contact_sheet"] = mistakes_saved
    common.write_json(output_json, evaluation)

    print(report_text)
    print(
        f"Test accuracy: {evaluation['metrics']['accuracy']:.4f} | "
        f"macro-F1: {evaluation['metrics']['macro_f1']:.4f} | "
        f"weighted-F1: {evaluation['metrics']['weighted_f1']:.4f}"
    )
    print(f"Wrote {output_json.relative_to(common.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
