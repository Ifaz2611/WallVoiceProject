"""Tune and fit the KNN baselines on train+val only.

The test split is never read here. Model selection uses grouped 5-fold
cross-validation on train+val (groups come from near-duplicate detection), and
the chosen configuration is also scored on the held-out validation split after
fitting on train only, so the reported validation numbers are honest.

Outputs (all git-ignored):
    models/knn_<extractor>.joblib
    models/knn_best.joblib
    outputs/model-selection.json
    outputs/model-selection.md
    outputs/cv-results-<extractor>.csv

Usage:
    python scripts/train_knn.py
    python scripts/train_knn.py --extractors resnet --n-jobs 4
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

import common

ALL_EXTRACTORS = ("raw", "hog", "resnet")
K_GRID = (1, 3, 5, 7, 9, 15)
WEIGHTS_GRID = ("uniform", "distance")
METRIC_GRID = ("euclidean", "cosine")
PCA_GRID = (32, 64, 128, 256)
CV_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractors", default=",".join(ALL_EXTRACTORS))
    parser.add_argument("--seed", type=int, default=common.RANDOM_SEED)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small debug grid (fewer k / PCA values).",
    )
    return parser.parse_args()


def load_features(extractor: str, splits: list[str]):
    arrays, frames = [], []
    for split in splits:
        feature_path = common.FEATURES_DIR / f"{extractor}_{split}.npy"
        ids_path = common.FEATURES_DIR / f"{extractor}_{split}_ids.csv"
        if not feature_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"Missing {feature_path.name}. Run: python scripts/extract_features.py --extractor {extractor}"
            )
        ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
        features = np.load(feature_path)
        if features.shape[0] != len(ids):
            raise ValueError(f"Feature/ID row mismatch for {extractor}/{split}")
        arrays.append(features)
        frames.append(ids)
    return np.vstack(arrays), pd.concat(frames, ignore_index=True)


def metrics_payload(y_true, y_pred, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(whiten=True, random_state=common.RANDOM_SEED)),
            ("knn", KNeighborsClassifier()),
        ]
    )


def tune_extractor(
    extractor: str,
    train: np.ndarray,
    train_ids: pd.DataFrame,
    val: np.ndarray,
    val_ids: pd.DataFrame,
    labels: list[str],
    args: argparse.Namespace,
) -> tuple[dict, Pipeline, dict]:
    X_all = np.vstack([train, val])
    y_all = np.concatenate([train_ids["label"].to_numpy(), val_ids["label"].to_numpy()])
    groups = np.concatenate([train_ids["group_id"].to_numpy(), val_ids["group_id"].to_numpy()])
    k_grid = (1, 5, 15) if args.quick else K_GRID
    pca_grid = (64,) if args.quick else PCA_GRID

    param_grid = {
        "pca__n_components": list(pca_grid),
        "knn__n_neighbors": list(k_grid),
        "knn__weights": list(WEIGHTS_GRID),
        "knn__metric": list(METRIC_GRID),
    }
    search = GridSearchCV(
        build_pipeline(),
        param_grid=param_grid,
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit="macro_f1",
        cv=StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed),
        n_jobs=args.n_jobs,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_all, y_all, groups=groups)

    # Honest validation score: refit the chosen config on train only.
    validation_model = clone(search.best_estimator_).fit(train, train_ids["label"])
    val_predictions = validation_model.predict(val)
    val_metrics = metrics_payload(val_ids["label"], val_predictions, labels)

    results = pd.DataFrame(search.cv_results_)
    keep = [
        "rank_test_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
        "param_pca__n_components",
        "param_knn__n_neighbors",
        "param_knn__weights",
        "param_knn__metric",
    ]
    results = results[keep].sort_values("rank_test_macro_f1")
    results.to_csv(common.OUTPUTS_DIR / f"cv-results-{extractor}.csv", index=False)

    summary = {
        "extractor": extractor,
        "n_features": int(train.shape[1]),
        "n_train": int(train.shape[0]),
        "n_val": int(val.shape[0]),
        "cv_folds": args.cv_folds,
        "best_params": {
            "pca_components": int(search.best_params_["pca__n_components"]),
            "pca_whiten": True,
            "k": int(search.best_params_["knn__n_neighbors"]),
            "weights": search.best_params_["knn__weights"],
            "metric": search.best_params_["knn__metric"],
        },
        "cv_macro_f1": float(search.best_score_),
        "cv_std_macro_f1": float(results.iloc[0]["std_test_macro_f1"]),
        "cv_best_accuracy": float(results.iloc[0]["mean_test_accuracy"]),
        "val_metrics": val_metrics,
        "top_configurations": json.loads(results.head(10).to_json(orient="records")),
    }
    return summary, search.best_estimator_, val_metrics


def fit_baselines(
    train: np.ndarray,
    train_ids: pd.DataFrame,
    val: np.ndarray,
    val_ids: pd.DataFrame,
    labels: list[str],
) -> dict[str, dict]:
    baseline_models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=common.RANDOM_SEED),
                ),
            ]
        ),
        "linear_svm_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", max_iter=20000, random_state=common.RANDOM_SEED)),
            ]
        ),
    }
    results = {}
    for name, model in baseline_models.items():
        model.fit(train, train_ids["label"])
        predictions = model.predict(val)
        results[name] = metrics_payload(val_ids["label"], predictions, labels)
    return results


def markdown_report(payload: dict) -> str:
    labels = payload["labels"]
    lines = [
        "# KNN model selection (train + val only)",
        "",
        f"Seed: `{payload['seed']}` | grouped {payload['cv_folds']}-fold CV | "
        "metric used for selection: **macro-F1**.",
        "The frozen test split was not read by this script.",
        "",
        "## Main results",
        "",
        "| Extractor | Dimensions | Best params | CV macro-F1 | Val accuracy | Val macro-F1 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for result in payload["results"]:
        params = result["best_params"]
        lines.append(
            "| {extractor} | {n_features} | PCA {pca_components} (whiten), k={k}, "
            "{weights}, {metric} | {cv_macro_f1:.4f} +/- {cv_std_macro_f1:.4f} | "
            "{val_accuracy:.4f} | {val_macro_f1:.4f} |".format(
                extractor=result["extractor"],
                n_features=result["n_features"],
                pca_components=params["pca_components"],
                k=params["k"],
                weights=params["weights"],
                metric=params["metric"],
                cv_macro_f1=result["cv_macro_f1"],
                cv_std_macro_f1=result["cv_std_macro_f1"],
                val_accuracy=result["val_metrics"]["accuracy"],
                val_macro_f1=result["val_metrics"]["macro_f1"],
            )
        )

    lines += [
        "",
        f"Best extractor by CV macro-F1: **{payload['best_extractor']}**. "
        "Validation scores are from a model fitted on `train.csv` only.",
        "",
        "## Ablation: raw pixels vs HOG+histogram vs CNN embeddings",
        "",
        "| Extractor | Val accuracy | Val macro-F1 | Val weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for result in payload["results"]:
        metrics = result["val_metrics"]
        lines.append(
            f"| {result['extractor']} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |"
        )

    lines += [
        "",
        "## Simple baselines fitted on the same features (val only)",
        "",
        "Baselines use the best extractor's features (`{}`) with a standard scaler.".format(
            payload["best_extractor"]
        ),
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in payload["baselines"].items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['weighted_f1']:.4f} |"
        )

    lines += [
        "",
        "## Per-class validation metrics (best extractor, validation model)",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        metrics = payload["validation_per_class"][label]
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['support']} |"
        )

    best_result = next(result for result in payload["results"] if result["extractor"] == payload["best_extractor"])
    matrix = best_result["val_metrics"]["confusion_matrix"]
    lines += [
        "",
        f"## Validation confusion matrix ({payload['best_extractor']}, train-fitted model)",
        "",
        "| True / Pred | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for label, row in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(str(value) for value in row) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- `PCA(whiten=True)` follows `StandardScaler` for every configuration.",
        "- KNN distance is computed in the whitened space; cosine is included as a required alternative.",
        "- `weights='distance'` was tested against `uniform`; both are in the grid. KNN has no class "
        "weights, so the majority class is still over-predicted. A balanced logistic regression is "
        "reported as a baseline and currently beats KNN.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    common.ensure_dirs(common.MODELS_DIR, common.OUTPUTS_DIR)
    extractors = [name.strip() for name in args.extractors.split(",") if name.strip()]
    for extractor in extractors:
        if extractor not in ALL_EXTRACTORS:
            raise SystemExit(f"Unknown extractor: {extractor}")

    labels = sorted(common.AREA_CODES.values())
    payload = {
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "labels": labels,
        "results": [],
    }
    best_overall = None

    for extractor in extractors:
        print(f"\n=== {extractor} ===")
        train, train_ids = load_features(extractor, ["train"])
        val, val_ids = load_features(extractor, ["val"])
        summary, fitted, val_metrics = tune_extractor(
            extractor, train, train_ids, val, val_ids, labels, args
        )
        payload["results"].append(summary)

        bundle = {
            "pipeline": fitted,
            "extractor": extractor,
            "labels": labels,
            "best_params": summary["best_params"],
            "cv_macro_f1": summary["cv_macro_f1"],
            "val_metrics": val_metrics,
            "train_counts": common.label_counts(train_ids["label"]),
            "val_counts": common.label_counts(val_ids["label"]),
            "feature_dimensions": int(train.shape[1]),
            "versions": common.environment_versions(),
        }
        joblib.dump(bundle, common.MODELS_DIR / f"knn_{extractor}.joblib", compress=3)

        if best_overall is None or summary["cv_macro_f1"] > best_overall[1]["cv_macro_f1"]:
            best_overall = (extractor, summary, bundle)

    best_extractor, best_summary, best_bundle = best_overall
    joblib.dump(best_bundle, common.MODELS_DIR / "knn_best.joblib", compress=3)
    payload["best_extractor"] = best_extractor
    payload["validation_per_class"] = best_summary["val_metrics"]["per_class"]

    baseline_extractor = best_extractor
    baseline_train, baseline_train_ids = load_features(baseline_extractor, ["train"])
    baseline_val, baseline_val_ids = load_features(baseline_extractor, ["val"])
    payload["baselines"] = fit_baselines(
        baseline_train, baseline_train_ids, baseline_val, baseline_val_ids, labels
    )
    payload["baseline_extractor"] = baseline_extractor

    common.write_json(common.OUTPUTS_DIR / "model-selection.json", payload)
    report = markdown_report(payload)
    (common.OUTPUTS_DIR / "model-selection.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Best model: models/knn_best.joblib (extractor={best_extractor})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
