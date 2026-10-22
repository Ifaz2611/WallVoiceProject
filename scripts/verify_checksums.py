"""Verify the original image archive against metadata/checksums.sha256.

Works on any OS (no `sha256sum` required). Exits non-zero on missing files or
hash mismatches.

Usage:
    python scripts/verify_checksums.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", default=str(common.ROOT / "metadata" / "checksums.sha256"))
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to image dataset directory (default: auto-detected or Dataset1)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_dir:
        common.set_dataset_dir(args.dataset_dir)
    checksum_path = Path(args.checksums)
    if not checksum_path.exists():
        raise SystemExit(f"Checksum file not found: {checksum_path}")

    checked = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split("  ", 1)
        path = common.image_path(relative_path)
        if not path.exists():
            missing.append(relative_path)
            continue
        if common.sha256_file(path) != digest:
            mismatched.append(relative_path)
        checked += 1

    print(f"Checked {checked} files from {checksum_path.relative_to(common.ROOT)}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for path in missing[:20]:
            print(f"  {path}")
    if mismatched:
        print(f"Hash mismatches ({len(mismatched)}):")
        for path in mismatched[:20]:
            print(f"  {path}")
    if missing or mismatched:
        return 1
    print("All checksums match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
