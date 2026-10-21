"""Shared helpers for the WallVoice24BD KNN baseline.

Everything in this module is deterministic given the image files and the
metadata CSV. Derived artefacts are written below ``outputs/`` (git-ignored)
so the original images are never modified.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "metadata" / "images.csv"
OUTPUTS_DIR = ROOT / "outputs"
AUDIT_DIR = OUTPUTS_DIR / "audit"
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
SPLITS_DIR = ROOT / "splits"

# Area folders and the image counts advertised in README.md.
EXPECTED_AREA_COUNTS = {
    "DU-Shahbagh": 428,
    "Dhanmondi": 312,
    "Azimpur-Nilkhet": 172,
    "BaileyRoad-Viqarunnisa": 136,
    "Buet-Palashi": 90,
    "Mirpur-Uttara-Tejgaon": 87,
}
EXPECTED_TOTAL = sum(EXPECTED_AREA_COUNTS.values())

# Area code used as the prediction label (see docs/label-guide.md).
AREA_CODES = {
    "DU-Shahbagh": "DUS",
    "Dhanmondi": "DMD",
    "Azimpur-Nilkhet": "AZN",
    "BaileyRoad-Viqarunnisa": "BRV",
    "Buet-Palashi": "BUP",
    "Mirpur-Uttara-Tejgaon": "MUT",
}
CODE_TO_AREA = {code: area for area, code in AREA_CODES.items()}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Duplicate / near-duplicate thresholds.
PHASH_SIZE = 32
PHASH_DISTANCE_THRESHOLD = 6  # Hamming distance on 64-bit pHash
CNN_COSINE_THRESHOLD = 0.93  # ResNet-18 ImageNet embedding cosine similarity

SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Metadata and paths
# ---------------------------------------------------------------------------
def load_metadata(path: Path = METADATA_PATH) -> pd.DataFrame:
    """Load the image inventory and normalise path separators."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "relative_path", "area"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df = df.copy()
    df["relative_path"] = df["relative_path"].str.replace("\\", "/", regex=False)
    for column in df.columns:
        df[column] = df[column].str.strip()
    return df


def image_path(relative_path: str) -> Path:
    return ROOT / relative_path


def open_rgb(path: str | Path) -> Image.Image:
    """Open an image, honour EXIF orientation, and convert to RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Hashing / near-duplicate detection
# ---------------------------------------------------------------------------
def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash_bits(image: Image.Image) -> np.ndarray:
    """64-bit DCT perceptual hash (packed into uint8[8])."""
    grey = np.asarray(
        image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    low = cv2.dct(grey)[:8, :8].flatten()
    median = float(np.median(low[1:]))
    return np.packbits(low > median)


def phash_hex(bits: np.ndarray) -> str:
    return "".join(f"{byte:02x}" for byte in bits.tolist())


def phash_from_hex(value: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8)


_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def hamming_matrix(packed: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed bit rows."""
    xor = np.bitwise_xor(packed[:, None, :], packed[None, :, :])
    return _POPCOUNT[xor].sum(axis=2).astype(np.int16)


def _file_fingerprint(df: pd.DataFrame) -> tuple[str, list[dict]]:
    rows = []
    for record in df.itertuples(index=False):
        path = image_path(record.relative_path)
        stat = path.stat()
        rows.append(
            {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = json.dumps(rows, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), rows


def load_or_compute_hashes(
    df: pd.DataFrame, cache_dir: Path = AUDIT_DIR
) -> tuple[pd.DataFrame, str]:
    """MD5 + pHash for every image, cached against file size/mtime."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = cache_dir / "hashes.csv"
    cache_meta = cache_dir / "hashes_meta.json"
    fingerprint, fingerprint_rows = _file_fingerprint(df)

    if cache_csv.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            cached = pd.read_csv(cache_csv, dtype=str, keep_default_na=False)
            if list(cached["image_id"]) == list(df["image_id"]):
                return cached, fingerprint

    records = []
    for record in df.itertuples(index=False):
        path = image_path(record.relative_path)
        with Image.open(path) as raw:
            bits = phash_bits(ImageOps.exif_transpose(raw))
        records.append(
            {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "md5": md5_file(path),
                "phash": phash_hex(bits),
                "file_size": path.stat().st_size,
            }
        )
    hashes = pd.DataFrame(records)
    hashes.to_csv(cache_csv, index=False)
    write_json(
        cache_meta,
        {"fingerprint": fingerprint, "files": fingerprint_rows},
    )
    return hashes, fingerprint


# ---------------------------------------------------------------------------
# CNN embeddings for near-duplicate detection
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _resnet_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _load_resnet18():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


def load_or_compute_embeddings(
    df: pd.DataFrame,
    cache_dir: Path = AUDIT_DIR,
    batch_size: int = 32,
    fingerprint: str | None = None,
) -> np.ndarray:
    """L2-normalised 512-d ResNet-18 embeddings used only for grouping."""
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npy = cache_dir / "resnet18_embedding.npy"
    cache_meta = cache_dir / "resnet18_embedding_meta.json"
    signature = f"resnet18-imagenet-224|{fingerprint or _file_fingerprint(df)[0]}"

    if cache_npy.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        if meta.get("signature") == signature:
            cached = np.load(cache_npy)
            if cached.shape == (len(df), 512):
                return cached

    model = _load_resnet18()
    transform = _resnet_transform()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            tensors = [transform(open_rgb(image_path(row.relative_path))) for row in chunk.itertuples()]
            embeddings = model(torch.stack(tensors))
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            batches.append(embeddings.cpu().numpy().astype(np.float32))
    matrix = np.vstack(batches)
    np.save(cache_npy, matrix)
    write_json(cache_meta, {"signature": signature, "shape": list(matrix.shape)})
    return matrix


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.reasons = [set() for _ in range(size)]

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, first: int, second: int, reason: str) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            self.reasons[root_a].add(reason)
            return
        # Keep the smaller index as root so the result is order independent.
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.reasons[root_a] |= self.reasons[root_b] | {reason}


def build_groups(
    df: pd.DataFrame,
    use_cnn: bool = True,
    cache_dir: Path = AUDIT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union exact duplicates, pHash near-duplicates and CNN near-duplicates.

    Returns ``(groups, candidate_pairs)``. A group is the leakage unit used by
    the split script; every image in a group must stay in one split.
    """
    n = len(df)
    hashes, fingerprint = load_or_compute_hashes(df, cache_dir)
    union_find = _UnionFind(n)
    pairs: list[dict] = []

    # 1. Exact duplicates (MD5).
    for _, indices in hashes.groupby("md5").groups.items():
        indices = list(indices)
        for other in indices[1:]:
            union_find.union(indices[0], other, "md5")
            pairs.append(
                {
                    "image_id_a": df.iloc[indices[0]].image_id,
                    "image_id_b": df.iloc[other].image_id,
                    "reason": "md5",
                    "score": 0.0,
                }
            )

    # 2. Perceptual-hash near-duplicates.
    packed = np.stack([phash_from_hex(value) for value in hashes["phash"]])
    distances = hamming_matrix(packed)
    for i, j in np.argwhere(np.triu(distances <= PHASH_DISTANCE_THRESHOLD, k=1)):
        union_find.union(int(i), int(j), "phash")
        pairs.append(
            {
                "image_id_a": df.iloc[i].image_id,
                "image_id_b": df.iloc[j].image_id,
                "reason": "phash",
                "score": float(distances[i, j]),
            }
        )

    # 3. CNN embedding near-duplicates (same wall / burst shots).
    if use_cnn:
        embeddings = load_or_compute_embeddings(df, cache_dir, fingerprint=fingerprint)
        similarities = embeddings @ embeddings.T
        np.fill_diagonal(similarities, -1.0)
        for i, j in np.argwhere(np.triu(similarities >= CNN_COSINE_THRESHOLD, k=1)):
            union_find.union(int(i), int(j), "cnn")
            pairs.append(
                {
                    "image_id_a": df.iloc[i].image_id,
                    "image_id_b": df.iloc[j].image_id,
                    "reason": "cnn",
                    "score": round(float(similarities[i, j]), 6),
                }
            )

    roots = [union_find.find(i) for i in range(n)]
    members: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        members.setdefault(root, []).append(index)

    ordered = sorted(members.values(), key=lambda idxs: min(df.iloc[i].image_id for i in idxs))
    group_ids = {tuple(idxs): f"G{number:04d}" for number, idxs in enumerate(ordered, start=1)}

    rows = []
    for index, root in enumerate(roots):
        members_root = members[root]
        areas = sorted({df.iloc[i].area for i in members_root})
        rows.append(
            {
                "image_id": df.iloc[index].image_id,
                "group_id": group_ids[tuple(members_root)],
                "group_size": len(members_root),
                "group_areas": ";".join(areas),
                "group_reasons": ";".join(sorted(union_find.reasons[root])),
            }
        )
    groups = pd.DataFrame(rows)
    pairs_df = pd.DataFrame(
        pairs,
        columns=["image_id_a", "image_id_b", "reason", "score"],
    )
    return groups, pairs_df


# ---------------------------------------------------------------------------
# Splits / features helpers
# ---------------------------------------------------------------------------
def read_split(name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/make_splits.py"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def label_counts(labels: Iterable[str]) -> dict[str, int]:
    counts = pd.Series(list(labels)).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}

    def add(name: str, module_name: str | None = None) -> None:
        try:
            module = __import__(module_name or name)
        except Exception:  # pragma: no cover - optional dependency
            return
        versions[name] = getattr(module, "__version__", "unknown")

    for name in ("numpy", "pandas", "sklearn", "PIL", "cv2", "matplotlib", "seaborn", "tqdm", "joblib"):
        add(name, {"sklearn": "sklearn", "PIL": "PIL"}.get(name, name))
    add("torch")
    add("torchvision")
    return versions


def write_checksums(path: Path | None = None) -> Path:
    """SHA-256 checksums for every original image (sorted by path)."""
    df = load_metadata()
    target = path or (ROOT / "metadata" / "checksums.sha256")
    lines = []
    for relative_path in sorted(df["relative_path"]):
        source = image_path(relative_path)
        if not source.exists():
            raise FileNotFoundError(source)
        lines.append(f"{sha256_file(source)}  {relative_path}")
    # Force LF so `sha256sum -c` works on Linux/macOS even when generated on Windows.
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return target
