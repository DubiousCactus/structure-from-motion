#!/usr/bin/env python3
"""Download the GGG SfM Benchmark dataset from Mendeley Data.

Downloads 271 images across 11 objects, organized into band-type subdirectories:
  three_bands_01/  (55 images, 360° × 3 elevations)
  three_bands_02/  (54 images)
  three_bands_03/  (54 images)
  one_band_01/     (18 images, 360° × 1 elevation)
  one_band_02/     (18 images)
  one_band_03/     (18 images)
  one_band_04/     (17 images)
  half_band_01/    (10 images, 180° × 1 elevation)
  half_band_02/    ( 9 images)
  half_band_03/    ( 9 images)
  half_band_04/    ( 9 images)

Usage:
    uv run python scripts/download_ggg.py
    uv run python scripts/download_ggg.py --output data/my_ggg --max-workers 16
"""

import argparse
import json
import shutil
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

API_URL = "https://data.mendeley.com/public-api/datasets/t4d8mv3fxt"

BAND_CLASSIFICATION = [
    (50, "three_bands"),
    (14, "one_band"),
    (0, "half_band"),
]


def classify_band(count: int) -> str:
    for threshold, name in BAND_CLASSIFICATION:
        if count >= threshold:
            return name
    return "half_band"


def download_file(file_info: dict, target_path: Path) -> dict | None:
    """Download a single file to *target_path* via atomic temp-file write.

    Returns a manifest entry on success, ``None`` on failure.
    """
    # File already on disk (e.g. from an interrupted-but-partially-done run where
    # the manifest was lost) — treat as success without re-downloading.
    if target_path.exists():
        return _manifest_entry(file_info, target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        req = urllib.request.Request(
            file_info["content_details"]["download_url"],
            headers={"User-Agent": "sfm-pipeline/0.1.0 (dataset download script)"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        tmp_path.rename(target_path)
        return _manifest_entry(file_info, target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return None


def _manifest_entry(file_info: dict, target_path: Path) -> dict:
    """Build a manifest entry for a successfully-downloaded file."""
    return {
        "path": str(target_path.relative_to(target_path.parent.parent)),
        "id": file_info["id"],
    }


def fetch_file_listing() -> list[dict]:
    """Fetch the full file list from the Mendeley API."""
    req = urllib.request.Request(
        API_URL,
        headers={
            "User-Agent": "sfm-pipeline/0.1.0 (dataset download script)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Failed to fetch dataset metadata: {e}", file=sys.stderr)
        sys.exit(1)

    files = data.get("files", [])
    if not files:
        print("No files found in the dataset.", file=sys.stderr)
        sys.exit(1)

    total_size = sum(f["content_details"]["size"] for f in files)
    print(
        f"Found {len(files)} files "
        f"({total_size // 1024 // 1024} MB total)"
    )
    return files


def build_download_plan(files: list[dict], out_dir: Path) -> list[tuple[dict, Path]]:
    """Build a list of ``(file_info, target_path)`` pairs.

    Image files are grouped by Mendeley *folder_id* and placed into band-type
    subdirectories (e.g., ``three_bands_01/``).  Non-image files
    (ReadMe.txt, CopyrightNotice.pdf) stay in the root *out_dir*.
    """
    meta_files = [f for f in files if "folder_id" not in f]
    image_files = [f for f in files if "folder_id" in f]

    groups: dict[str, list[dict]] = defaultdict(list)
    for f in image_files:
        groups[f["folder_id"]].append(f)

    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    counters: dict[str, int] = defaultdict(int)
    dir_map: dict[str, str] = {}
    for folder_id, flist in sorted_groups:
        band = classify_band(len(flist))
        counters[band] += 1
        dir_map[folder_id] = f"{band}_{counters[band]:02d}"

    plan: list[tuple[dict, Path]] = []
    for f in meta_files:
        plan.append((f, out_dir / f["filename"]))
    for f in image_files:
        plan.append((f, out_dir / dir_map[f["folder_id"]] / f["filename"]))

    return plan


def main(output_dir: str, max_workers: int = 8) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = fetch_file_listing()
    plan = build_download_plan(files, out_dir)

    manifest_path = out_dir / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing_manifest = json.load(f)
        existing_ids = {e["id"] for e in existing_manifest}
    else:
        existing_manifest = []
        existing_ids = set()

    manifest = list(existing_manifest)
    to_download = [(fi, tp) for fi, tp in plan if fi["id"] not in existing_ids]

    if not to_download:
        print("All files already downloaded.")
        return

    print(f"Downloading {len(to_download)} files with {max_workers} workers ...")

    n_ok = 0
    n_fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_file, file_info, target_path): file_info["id"]
            for file_info, target_path in to_download
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading", unit="file"
        ):
            result = future.result()
            if result is not None:
                manifest.append(result)
                n_ok += 1
            else:
                n_fail += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(f["content_details"]["size"] for f in files)
    msg = (
        f"Done — {len(manifest)} / {len(files)} files "
        f"({total_size // 1024 // 1024} MB) → {out_dir.resolve()}"
    )
    if n_fail:
        msg += f", {n_fail} failed"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GGG SfM Benchmark dataset")
    parser.add_argument(
        "--output",
        default="data/ggg",
        help="Output directory (default: data/ggg)",
    )
    parser.add_argument(
        "--max-workers",
        default=8,
        type=int,
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()
    main(args.output, max_workers=args.max_workers)
