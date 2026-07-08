#!/usr/bin/env python3
"""Download the SFMedu benchmark images from GitHub.

Images: B21.jpg – B25.jpg (5 images)

Output:
    data/sfmedu/
        B21.jpg
        ...

Usage:
    uv run python scripts/download_sfmedu.py
    uv run python scripts/download_sfmedu.py --output data/sfmedu
"""

import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm

RAW = "https://raw.githubusercontent.com/jianxiongxiao/SFMedu/master/images"
FILES = [f"B2{i}.jpg" for i in range(1, 6)]


def download_file(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "sfm-pipeline/0.1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk_size = 8 * 1024 * 1024
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            while chunk := resp.read(chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)


def run(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = {f.name for f in out.iterdir()} if out.exists() else set()
    missing = [f for f in FILES if f not in existing]

    if not missing:
        print("All files already downloaded.")
        return

    print(f"Downloading {len(missing)} file(s) to {out.resolve()} ...")
    for name in missing:
        download_file(f"{RAW}/{name}", out / name)

    print(f"Done — {len(missing)} / {len(FILES)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SFMedu benchmark images")
    parser.add_argument(
        "--output",
        default="data/sfmedu",
        help="Output directory (default: data/sfmedu)",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
