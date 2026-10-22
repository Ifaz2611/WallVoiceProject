"""Extract deterministic image features for the WallVoice24BD KNN baseline.

Three extractors are available:

* ``raw``   -- 32x32 RGB pixels (plain baseline / ablation).
* ``hog``   -- 128x128 colour histograms (RGB + HSV) concatenated with OpenCV HOG.
* ``resnet``-- 224x224 ResNet-18 ImageNet penultimate embeddings (512-d, L2-normalised).

Features are cached to ``features/<extractor>_<split>.npy`` together with an
ID file, and extractor settings plus library versions go to
``features/config.json``.

Usage:
    python scripts/extract_features.py --extractor all
    python scripts/extract_features.py --extractor resnet --force
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import common

RAW_SIZE = 32
HOG_SIZE = 128
RESNET_SIZE = 224

EXTRACTOR_DESCRIPTIONS = {
    "raw": {
        "type": "pixels",
        "resize": [RAW_SIZE, RAW_SIZE],
        "description": "32x32 RGB pixels scaled to [0, 1], flattened.",
    },
    "hog": {
        "type": "handcrafted",
        "resize": [HOG_SIZE, HOG_SIZE],
        "description": (
            "RGB histogram (32 bins/channel, 96-d) + HSV H-S histogram "
            "(16x8, 128-d) + OpenCV HOG (16x16 cells, 2x2 blocks, 9 orientations)."
        ),
    },
    "resnet": {
        "type": "cnn",
        "backbone": "resnet18",
        "weights": "torchvision ImageNet DEFAULT",
        "resize": [RESNET_SIZE, RESNET_SIZE],
        "description": "ResNet-18 penultimate activations (512-d), L2-normalised.",
    },
}

_HOG = cv2.HOGDescriptor(
    _winSize=(HOG_SIZE, HOG_SIZE),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extractor",
        choices=["raw", "hog", "resnet", "all"],
        default="all",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force", action="store_true", help="Recompute cached features.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to image dataset directory (default: auto-detected or Dataset1)",
    )
    return parser.parse_args()


def raw_features(image) -> np.ndarray:
    resized = np.asarray(
        image.resize((RAW_SIZE, RAW_SIZE), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return (resized / 255.0).flatten()


def hog_features(image) -> np.ndarray:
    resized = np.asarray(
        image.resize((HOG_SIZE, HOG_SIZE), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    rgb_hist = np.concatenate(
        [cv2.calcHist([resized], [channel], None, [32], [0, 256]).ravel() for channel in range(3)]
    )
    rgb_hist = rgb_hist / max(float(rgb_hist.sum()), 1.0)

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    hs_hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    hs_hist = hs_hist / max(float(hs_hist.sum()), 1.0)

    hog = _HOG.compute(resized).ravel()
    features = np.concatenate([rgb_hist, hs_hist, hog]).astype(np.float32)
    return normalize_l2(features)


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return (vector / norm).astype(np.float32)


def resnet_extractor():
    """Return (model, transform) for ResNet-18 embeddings."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    return model, transform, torch


def extract_split(
    extractor: str,
    split: str,
    ids: pd.DataFrame,
    force: bool,
    resnet_bundle=None,
) -> tuple[np.ndarray, int]:
    feature_path = common.FEATURES_DIR / f"{extractor}_{split}.npy"
    ids_path = common.FEATURES_DIR / f"{extractor}_{split}_ids.csv"

    if feature_path.exists() and ids_path.exists() and not force:
        cached_ids = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
        if list(cached_ids["image_id"]) == list(ids["image_id"]):
            features = np.load(feature_path)
            print(f"  {extractor}/{split}: cached {features.shape}")
            return features, features.shape[1]

    vectors = []
    if extractor == "resnet":
        if resnet_bundle is None:
            resnet_bundle = resnet_extractor()
        model, transform, torch = resnet_bundle
        tensors = []
        with torch.no_grad():
            for start in range(0, len(ids), 32):
                chunk = ids.iloc[start : start + 32]
                tensors = [
                    transform(common.open_rgb(common.image_path(row.relative_path)))
                    for row in chunk.itertuples()
                ]
                batch = model(torch.stack(tensors))
                batch = torch.nn.functional.normalize(batch, p=2, dim=1)
                vectors.append(batch.cpu().numpy().astype(np.float32))
        features = np.vstack(vectors)
    else:
        function = raw_features if extractor == "raw" else hog_features
        for row in tqdm(ids.itertuples(index=False), total=len(ids), desc=f"{extractor}/{split}"):
            image = common.open_rgb(common.image_path(row.relative_path))
            vectors.append(function(image))
        features = np.vstack(vectors).astype(np.float32)

    np.save(feature_path, features)
    ids.to_csv(ids_path, index=False)
    print(f"  {extractor}/{split}: {features.shape}")
    return features, features.shape[1]


def main() -> int:
    args = parse_args()
    if args.dataset_dir:
        common.set_dataset_dir(args.dataset_dir)
    common.ensure_dirs(common.FEATURES_DIR)
    splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    extractors = ["raw", "hog", "resnet"] if args.extractor == "all" else [args.extractor]

    config_path = common.FEATURES_DIR / "config.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
    config.setdefault("extractors", {})

    split_frames = {split: common.read_split(split) for split in splits}
    dimensions: dict[str, dict[str, int]] = {}

    for extractor in extractors:
        print(f"Extracting {extractor}")
        bundle = resnet_extractor() if extractor == "resnet" else None
        dims = {}
        for split in splits:
            _, dim = extract_split(extractor, split, split_frames[split], args.force, bundle)
            dims[split] = int(dim)
        dimensions[extractor] = dims
        config["extractors"][extractor] = {
            **EXTRACTOR_DESCRIPTIONS[extractor],
            "dimension": dims,
        }

    config.update(
        {
            "versions": common.environment_versions(),
            "dimensions": {**config.get("dimensions", {}), **dimensions},
            "labels": sorted(
                set().union(*(set(frame["label"]) for frame in split_frames.values()))
            ),
            "splits": {
                split: {
                    "images": int(len(frame)),
                    "labels": common.label_counts(frame["label"]),
                }
                for split, frame in split_frames.items()
            },
            "normalisation": (
                "ResNet-18 embeddings and HOG+histogram vectors are L2-normalised after "
                "extraction; raw pixels are scaled to [0, 1]."
            ),
        }
    )
    common.write_json(config_path, config)
    print(f"Wrote {config_path.relative_to(common.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
