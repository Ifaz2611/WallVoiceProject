"""Sanity-check feature quality with PCA + t-SNE coloured by area.

If the six areas overlap heavily even in a nonlinear projection, fix the
features before tuning KNN further. Saves a two-panel figure and the
silhouette scores to ``outputs/tsne-features.png`` / ``outputs/tsne-scores.json``.

Usage:
    python scripts/visualize_features.py
    python scripts/visualize_features.py --extractors hog,resnet
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common

COLOURS = {
    "AZN": "#1f77b4",
    "BRV": "#ff7f0e",
    "BUP": "#2ca02c",
    "DMD": "#d62728",
    "DUS": "#9467bd",
    "MUT": "#8c564b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractors", default="hog,resnet")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=common.RANDOM_SEED)
    return parser.parse_args()


def load_all(extractor: str) -> tuple[np.ndarray, pd.DataFrame]:
    features, frames = [], []
    for split in common.SPLIT_NAMES:
        feature_path = common.FEATURES_DIR / f"{extractor}_{split}.npy"
        ids_path = common.FEATURES_DIR / f"{extractor}_{split}_ids.csv"
        if not feature_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"Missing {feature_path.name}. Run scripts/extract_features.py first."
            )
        features.append(np.load(feature_path))
        frames.append(pd.read_csv(ids_path, dtype=str, keep_default_na=False))
    return np.vstack(features), pd.concat(frames, ignore_index=True)


def project(
    features: np.ndarray, labels: list[str], seed: int, perplexity: float
) -> tuple[np.ndarray, float]:
    scaled = StandardScaler().fit_transform(features)
    components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    reduced = PCA(n_components=components, random_state=seed).fit_transform(scaled)
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        n_jobs=-1,
    ).fit_transform(reduced)
    score = float(silhouette_score(reduced, labels, metric="euclidean"))
    return embedding, score


def main() -> int:
    args = parse_args()
    common.ensure_dirs(common.OUTPUTS_DIR)
    extractors = [name.strip() for name in args.extractors.split(",") if name.strip()]
    figure, axes = plt.subplots(1, len(extractors), figsize=(7 * len(extractors), 6))
    axes = np.atleast_1d(axes)
    scores = {}

    for axis, extractor in zip(axes, extractors):
        features, ids = load_all(extractor)
        embedding, score = project(features, ids["label"].tolist(), args.seed, args.perplexity)
        scores[extractor] = score
        for label in sorted(COLOURS):
            mask = ids["label"].to_numpy() == label
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=9,
                alpha=0.75,
                color=COLOURS[label],
                label=label,
            )
        axis.set_title(f"{extractor}: PCA(50) + t-SNE\nsilhouette (PCA space) = {score:.3f}")
        axis.set_xticks([])
        axis.set_yticks([])

    axes[-1].legend(markerscale=2, fontsize=8, frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(common.OUTPUTS_DIR / "tsne-features.png", dpi=150)
    plt.close(figure)
    common.write_json(common.OUTPUTS_DIR / "tsne-scores.json", scores)
    print(json.dumps(scores, indent=2))
    print(f"Wrote {(common.OUTPUTS_DIR / 'tsne-features.png').relative_to(common.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
