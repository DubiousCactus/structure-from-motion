#!/usr/bin/env python3
"""Download the Mip-NeRF 360 dataset (360_v2 split, 9 scenes, ~12 GB).

Scenes: garden, bicycle, counter, kitchen, room, stump, bonsai, flowers, treehill

Output structure:
    data/mipnerf360/
        garden/
            images/      (IMG_*.jpg)
            sparse/0/    (COLMAP cameras.bin, images.bin, points3D.bin)
        bicycle/
            ...

Usage:
    uv run python scripts/download_mipnerf360.py
    uv run python scripts/download_mipnerf360.py --output data/mipnerf360 --scenes garden bicycle
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ZIP_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

SCENES = [
    "garden",
    "bicycle",
    "counter",
    "kitchen",
    "room",
    "stump",
    "bonsai",
    "flowers",
    "treehill",
]


def _scene_complete(scene_dir: Path) -> bool:
    """Check if *scene_dir/images/* exists and contains at least one file."""
    images_dir = scene_dir / "images"
    return images_dir.exists() and any(images_dir.iterdir())


def _download_zip(url: str, dest: Path) -> None:
    """Stream *url* to *dest* with a tqdm progress bar, using .tmp staging."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    tmp.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={
        "User-Agent": "sfm-pipeline/0.1.0 (dataset download script)",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            chunk_size = 8 * 1024 * 1024  # 8 MiB
            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as pbar:
                while chunk := resp.read(chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        tmp.rename(dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _extract_scenes(zip_path: Path, scenes: set[str], out_dir: Path) -> None:
    """Extract entries matching *scenes* from *zip_path*.

    The zip has scene directories at the top level (e.g. ``bicycle/``,
    ``garden/``).  Some entries like ``flowers.txt`` are placeholders
    rather than actual scenes — we skip them.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        # Filter to entries under requested scene directories
        prefixes = {f"{s}/" for s in scenes}
        entries = [(n, n) for n in names if any(n.startswith(p) for p in prefixes)]

        if not entries:
            print("  No matching entries found in zip.", file=sys.stderr)
            return

        pbar = tqdm(entries, desc="Extracting", unit="file")
        for name, rel in pbar:
            target = out_dir / rel
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            pbar.set_postfix_str(rel, refresh=False)


def run(output_dir: str, scenes: list[str] | None = None) -> None:
    out = Path(output_dir)
    scenes_set = set(scenes or SCENES)

    # Check which scenes are already on disk
    complete = {s: _scene_complete(out / s) for s in scenes_set}
    missing = [s for s in scenes_set if not complete[s]]

    if not missing:
        print("All requested scenes already downloaded.")
        return

    print(f"Scenes to download: {', '.join(missing)}")

    zip_path = out / "360_v2.zip"
    if not zip_path.exists():
        _download_zip(ZIP_URL, zip_path)
    else:
        print("Using cached 360_v2.zip")

    _extract_scenes(zip_path, set(missing), out)

    n_ok = sum(1 for s in missing if _scene_complete(out / s))
    n_fail = len(missing) - n_ok
    msg = f"Done — {n_ok} / {len(missing)} scenes extracted to {out.resolve()}"
    if n_fail:
        msg += f", {n_fail} incomplete"
        # Failed scenes will be re-extracted on the next run (idempotent).
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Mip-NeRF 360 dataset (360_v2)"
    )
    parser.add_argument(
        "--output",
        default="data/mipnerf360",
        help="Output directory (default: data/mipnerf360)",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        choices=SCENES,
        metavar="SCENE",
        default=None,
        help=f"Scenes to download (default: all {len(SCENES)})",
    )
    args = parser.parse_args()
    run(args.output, scenes=args.scenes)


if __name__ == "__main__":
    main()
