"""Audit the WallVoice24BD image inventory.

Checks that every metadata row resolves to a readable image, that counts match
README.md, and that exact / near-duplicate images are identified before any
modelling happens. Writes derived reports to ``outputs/audit/`` and never
modifies the originals.

Usage:
    python scripts/audit.py
    python scripts/audit.py --skip-cnn      # faster, exact + pHash only
    python scripts/audit.py --checksums     # also write metadata/checksums.sha256
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip ResNet-18 near-duplicate detection (keeps exact MD5 + pHash).",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="Write SHA-256 checksums to metadata/checksums.sha256.",
    )
    parser.add_argument(
        "--sample-sheets",
        action="store_true",
        help="Write a contact sheet of 8 random images per area for manual review.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to image dataset directory (default: auto-detected or Dataset1)",
    )
    return parser.parse_args()


def write_sample_sheets(df: pd.DataFrame, per_area: int = 8) -> list[Path]:
    """Write review contact sheets under outputs/audit/sample-sheets/."""
    directory = common.AUDIT_DIR / "sample-sheets"
    directory.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = 240, 180
    columns = 4
    rows = (per_area + columns - 1) // columns
    written = []
    for area, subset in df.groupby("area", sort=True):
        sample = subset.sample(n=min(per_area, len(subset)), random_state=common.RANDOM_SEED)
        sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + 16)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (_, record) in enumerate(sample.iterrows()):
            image = common.open_rgb(common.image_path(record.relative_path))
            thumbnail = ImageOps.fit(image, (cell_width, cell_height))
            x = (index % columns) * cell_width
            y = (index // columns) * (cell_height + 16)
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 4, y + cell_height + 2), record.image_id, fill="black")
        path = directory / f"{area}.png"
        sheet.save(path)
        written.append(path)
    return written


def inspect_image(path: Path) -> dict:
    """Open an image and collect structural information without changing it."""
    try:
        with Image.open(path) as raw:
            raw.verify()
        with Image.open(path) as raw:
            width, height = raw.size
            mode = raw.mode
            image_format = raw.format
            orientation = raw.getexif().get(274, 1)
        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "format": image_format,
            "exif_orientation": orientation,
            "error": "",
        }
    except Exception as error:  # noqa: BLE001 - report every unreadable file
        return {
            "valid": False,
            "width": "",
            "height": "",
            "mode": "",
            "format": "",
            "exif_orientation": "",
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    args = parse_args()
    if args.dataset_dir:
        common.set_dataset_dir(args.dataset_dir)
    common.ensure_dirs(common.AUDIT_DIR)

    df = common.load_metadata()
    rows = []
    for record in tqdm(df.itertuples(index=False), total=len(df), desc="inspecting"):
        path = common.image_path(record.relative_path)
        info = {
            "image_id": record.image_id,
            "relative_path": record.relative_path,
            "area": record.area,
            "exists": path.exists(),
        }
        info.update(inspect_image(path) if path.exists() else {"valid": False, "error": "missing file"})
        rows.append(info)
    audit = pd.DataFrame(rows)

    # Files on disk that are not listed in metadata.
    listed = set(df["relative_path"])
    extra_files = []
    candidate_roots = [common.DATASET_DIR]
    if common.ROOT != common.DATASET_DIR:
        candidate_roots.append(common.ROOT)

    for base_dir in candidate_roots:
        if not base_dir.exists():
            continue
        for area in common.EXPECTED_AREA_COUNTS:
            folder = base_dir / area
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in common.IMAGE_EXTENSIONS:
                    rel = common.relative(path)
                    if rel not in listed and rel not in extra_files:
                        extra_files.append(rel)

    # README consistency and metadata integrity.
    area_counts = df["area"].value_counts().to_dict()
    count_mismatches = {
        area: {"readme": expected, "metadata": int(area_counts.get(area, 0))}
        for area, expected in common.EXPECTED_AREA_COUNTS.items()
        if int(area_counts.get(area, 0)) != expected
    }
    duplicate_ids = df["image_id"][df["image_id"].duplicated()].tolist()
    duplicate_paths = df["relative_path"][df["relative_path"].duplicated()].tolist()

    # Duplicate / near-duplicate groups (only readable images participate).
    valid_df = df.merge(
        audit.loc[audit["valid"], ["image_id"]], on="image_id", how="inner"
    )
    groups = pd.DataFrame(columns=["image_id", "group_id", "group_size", "group_areas", "group_reasons"])
    pairs = pd.DataFrame(columns=["image_id_a", "image_id_b", "reason", "score"])
    grouping_error = ""
    if len(valid_df):
        try:
            groups, pairs = common.build_groups(valid_df, use_cnn=not args.skip_cnn)
        except Exception as error:  # noqa: BLE001 - audit should still produce a report
            grouping_error = f"{type(error).__name__}: {error}"

    audit = audit.merge(groups, on="image_id", how="left")
    audit["label"] = audit["area"].map(common.AREA_CODES)
    audit["group_id"] = audit["group_id"].fillna("")
    audit["group_size"] = audit["group_size"].fillna(1).astype(int)
    audit["group_areas"] = audit["group_areas"].fillna("")
    audit["group_reasons"] = audit["group_reasons"].fillna("")

    # Attach hashes when they were cached / computed.
    hashes_path = common.AUDIT_DIR / "hashes.csv"
    if hashes_path.exists():
        hashes = pd.read_csv(hashes_path, dtype=str, keep_default_na=False)
        audit = audit.merge(hashes[["image_id", "md5", "phash"]], on="image_id", how="left")

    audit.to_csv(common.AUDIT_DIR / "images_audit.csv", index=False)
    if len(pairs):
        pairs = pairs.merge(
            df[["image_id", "relative_path", "area"]].rename(
                columns={"relative_path": "relative_path_a", "area": "area_a"}
            ),
            left_on="image_id_a",
            right_on="image_id",
            how="left",
        ).drop(columns="image_id")
        pairs = pairs.merge(
            df[["image_id", "relative_path", "area"]].rename(
                columns={"relative_path": "relative_path_b", "area": "area_b"}
            ),
            left_on="image_id_b",
            right_on="image_id",
            how="left",
        ).drop(columns="image_id")
        pairs.to_csv(common.AUDIT_DIR / "duplicate_pairs.csv", index=False)
    group_table = audit.loc[audit["group_size"] > 1, [
        "image_id",
        "relative_path",
        "area",
        "label",
        "group_id",
        "group_size",
        "group_areas",
        "group_reasons",
    ]].sort_values(["group_id", "image_id"])
    group_table.to_csv(common.AUDIT_DIR / "near_duplicate_groups.csv", index=False)

    multi_groups = group_table["group_id"].nunique() if len(group_table) else 0
    largest_group = int(group_table.groupby("group_id").size().max()) if len(group_table) else 1
    cross_area_groups = int(
        group_table.groupby("group_id")["group_areas"]
        .first()
        .str.contains(";")
        .sum()
    ) if len(group_table) else 0
    reason_counts = Counter(
        reason
        for value in group_table["group_reasons"]
        for reason in str(value).split(";")
        if reason
    )

    summary = {
        "dataset_version": "WallVoice24BD as committed in this repository",
        "metadata_rows": int(len(df)),
        "unique_image_ids": int(df["image_id"].nunique()),
        "unique_paths": int(df["relative_path"].nunique()),
        "expected_total": common.EXPECTED_TOTAL,
        "counts_match_readme": not count_mismatches and len(df) == common.EXPECTED_TOTAL,
        "count_mismatches": count_mismatches,
        "duplicate_image_ids": duplicate_ids,
        "duplicate_paths": duplicate_paths,
        "missing_files": audit.loc[~audit["exists"], "relative_path"].tolist(),
        "unreadable_files": audit.loc[audit["exists"] & ~audit["valid"], ["relative_path", "error"]].to_dict("records"),
        "extra_files_not_in_metadata": extra_files,
        "modes": dict(sorted(Counter(audit.loc[audit["valid"], "mode"]).items())),
        "formats": dict(sorted(Counter(audit.loc[audit["valid"], "format"]).items())),
        "exif_orientation_nonstandard": int((audit["exif_orientation"].astype(str) != "1").sum()),
        "min_width": int(pd.to_numeric(audit["width"], errors="coerce").min()),
        "min_height": int(pd.to_numeric(audit["height"], errors="coerce").min()),
        "images_smaller_than_256": int(
            (
                (pd.to_numeric(audit["width"], errors="coerce") < 256)
                | (pd.to_numeric(audit["height"], errors="coerce") < 256)
            ).sum()
        ),
        "label_counts": common.label_counts(df["area"].map(common.AREA_CODES)),
        "review_status_counts": df["review_status"].replace("", "unreviewed").value_counts().to_dict()
        if "review_status" in df
        else {},
        "exact_duplicate_pairs": int((pairs["reason"] == "md5").sum()) if len(pairs) else 0,
        "phash_near_duplicate_pairs": int((pairs["reason"] == "phash").sum()) if len(pairs) else 0,
        "cnn_near_duplicate_pairs": int((pairs["reason"] == "cnn").sum()) if len(pairs) else 0,
        "grouping_used_cnn": not args.skip_cnn,
        "grouping_error": grouping_error,
        "groups": int(audit["group_id"].replace("", pd.NA).nunique()),
        "multi_image_groups": multi_groups,
        "largest_group": largest_group,
        "cross_area_groups": cross_area_groups,
        "group_reason_counts": dict(reason_counts),
        "thresholds": {
            "phash_hamming": common.PHASH_DISTANCE_THRESHOLD,
            "cnn_cosine": common.CNN_COSINE_THRESHOLD,
        },
    }
    common.write_json(common.AUDIT_DIR / "summary.json", summary)

    if args.checksums:
        checksum_path = common.write_checksums()
        print(f"Wrote {checksum_path.relative_to(common.ROOT)}")

    if args.sample_sheets:
        sheets = write_sample_sheets(df)
        print(f"Wrote {len(sheets)} sample sheets to {(common.AUDIT_DIR / 'sample-sheets').relative_to(common.ROOT)}")

    printable = {
        key: summary[key]
        for key in (
            "metadata_rows",
            "unique_image_ids",
            "counts_match_readme",
            "missing_files",
            "unreadable_files",
            "extra_files_not_in_metadata",
            "modes",
            "exif_orientation_nonstandard",
            "images_smaller_than_256",
            "exact_duplicate_pairs",
            "phash_near_duplicate_pairs",
            "cnn_near_duplicate_pairs",
            "groups",
            "multi_image_groups",
            "largest_group",
            "cross_area_groups",
            "label_counts",
            "grouping_error",
        )
    }
    print(json.dumps(printable, indent=2))
    print(f"\nReports: {(common.AUDIT_DIR / 'summary.json').relative_to(common.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
