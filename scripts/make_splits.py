"""Create leakage-aware train/validation/test splits.

Images are grouped by exact duplicate / pHash / ResNet-18 near-duplicate
evidence (see ``scripts/common.py``). Whole groups are assigned to a split, so
a wall or near-duplicate view can never appear in two splits. Groups are
stratified by area to keep approximately 70/15/15 per area.

The test split is frozen: re-running the script refuses to overwrite existing
split files unless ``--force`` is passed, and with the same data and seed the
output is identical anyway.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --force
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import pandas as pd

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--seed", type=int, default=common.RANDOM_SEED)
    return parser.parse_args()


def assign_split(
    group_sizes: dict[str, int],
    group_area: dict[str, str],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, keeping per-area proportions close to target.

    Greedy on the largest relative deficit: each group goes to the split that is
    furthest behind its proportional target for that area. Shuffling first keeps
    the result deterministic but not tied to filename order.
    """
    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    by_area: dict[str, list[str]] = defaultdict(list)
    for group_id, area in group_area.items():
        by_area[area].append(group_id)

    for area in sorted(by_area):
        groups = by_area[area]
        rng.shuffle(groups)
        groups.sort(key=lambda group_id: -group_sizes[group_id])  # large groups first
        area_total = sum(group_sizes[group_id] for group_id in groups)
        targets = {split: common.SPLIT_FRACTIONS[split] * area_total for split in common.SPLIT_NAMES}
        counts = {split: 0 for split in common.SPLIT_NAMES}

        # Deterministic tie-break order per area.
        tie_order = list(common.SPLIT_NAMES)
        rng.shuffle(tie_order)

        for group_id in groups:
            split = max(
                tie_order,
                key=lambda name: ((targets[name] - counts[name]) / max(targets[name], 1.0), -tie_order.index(name)),
            )
            assignment[group_id] = split
            counts[split] += group_sizes[group_id]

    return assignment


def main() -> int:
    args = parse_args()
    common.ensure_dirs(common.SPLITS_DIR, common.AUDIT_DIR)

    existing = [common.SPLITS_DIR / f"{name}.csv" for name in common.SPLIT_NAMES]
    if any(path.exists() for path in existing) and not args.force:
        raise SystemExit(
            "Split files already exist and the test split is frozen. "
            "Pass --force only if you intentionally rebuild all splits."
        )

    df = common.load_metadata()
    df["label"] = df["area"].map(common.AREA_CODES)
    if df["label"].isna().any():
        unknown = sorted(df.loc[df["label"].isna(), "area"].unique())
        raise SystemExit(f"Unknown area folder(s) in metadata: {unknown}")

    print("Grouping near-duplicate images (MD5 + pHash + ResNet-18)...")
    groups, pairs = common.build_groups(df, use_cnn=True)

    df = df.merge(groups, on="image_id", how="left", validate="one_to_one")
    missing_groups = int(df["group_id"].isna().sum())
    if missing_groups:
        raise SystemExit(f"{missing_groups} images could not be grouped; run scripts/audit.py")

    group_table = (
        df.groupby("group_id", as_index=False)
        .agg(size=("image_id", "size"), area=("area", "first"), label=("label", "first"))
    )
    group_sizes = dict(zip(group_table["group_id"], group_table["size"]))
    group_area = dict(zip(group_table["group_id"], group_table["area"]))

    assignment = assign_split(group_sizes, group_area, args.seed)
    df["split"] = df["group_id"].map(assignment)

    # Hard leakage checks before writing anything.
    assert df["split"].notna().all(), "Every image must receive a split"
    grouped = df.groupby("group_id")["split"].nunique()
    if int(grouped.max()) != 1:
        raise SystemExit("Leakage check failed: a near-duplicate group spans multiple splits")

    if int(df.groupby("image_id")["split"].nunique().max()) != 1:
        raise SystemExit("Leakage check failed: an image appears in multiple splits")

    output_columns = ["image_id", "relative_path", "label", "group_id", "area"]
    for split in common.SPLIT_NAMES:
        subset = df.loc[df["split"] == split, output_columns].sort_values("image_id")
        subset.to_csv(common.SPLITS_DIR / f"{split}.csv", index=False, lineterminator="\n")

    # Recompute overlap after writing to prove near-duplicate groups are intact.
    written = {split: common.read_split(split) for split in common.SPLIT_NAMES}
    groups_per_split = {split: set(part["group_id"]) for split, part in written.items()}
    overlaps = {
        "train_val": len(groups_per_split["train"] & groups_per_split["val"]),
        "train_test": len(groups_per_split["train"] & groups_per_split["test"]),
        "val_test": len(groups_per_split["val"] & groups_per_split["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"Leakage check failed after writing: {overlaps}")

    summary = {
        "seed": args.seed,
        "split_fractions": common.SPLIT_FRACTIONS,
        "grouping": {
            "phash_hamming_threshold": common.PHASH_DISTANCE_THRESHOLD,
            "cnn_cosine_threshold": common.CNN_COSINE_THRESHOLD,
            "groups": int(group_table.shape[0]),
            "multi_image_groups": int((group_table["size"] > 1).sum()),
            "largest_group": int(group_table["size"].max()),
            "duplicate_pair_candidates": int(len(pairs)),
        },
        "counts": {split: int(len(part)) for split, part in written.items()},
        "per_area": {
            area: {
                split: int(((df["area"] == area) & (df["split"] == split)).sum())
                for split in common.SPLIT_NAMES
            }
            for area in common.EXPECTED_AREA_COUNTS
        },
        "labels": {
            split: common.label_counts(part["label"]) for split, part in written.items()
        },
        "group_overlap": overlaps,
        "no_group_spans_splits": bool(int(grouped.max()) == 1),
    }
    common.write_json(common.SPLITS_DIR / "summary.json", summary)
    df[["image_id", "relative_path", "label", "group_id", "split"]].sort_values("image_id").to_csv(
        common.SPLITS_DIR / "groups.csv", index=False, lineterminator="\n"
    )

    print(json.dumps({key: summary[key] for key in ("counts", "per_area", "group_overlap")}, indent=2))
    print(f"\nWrote {len(common.SPLIT_NAMES)} split files to splits/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
