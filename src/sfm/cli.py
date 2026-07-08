import os
from typing import Optional

import typer

from sfm.feature_extraction import extract_and_match_impl, extract_frames_impl

app = typer.Typer()


@app.command()
def extract_frames(video_path: str, output_folder: str):
    extract_frames_impl(video_path, output_folder)


@app.command()
def extract_and_match(
    frames_path: str,
    intrinsics_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    debug: Optional[bool] = False,
):
    extract_and_match_impl(frames_path, intrinsics_path, max_frames, debug)


@app.command(name="extract-intrinsics")
def extract_intrinsics(
    images_path: str,
    output: str = "intrinsics.npz",
):
    from sfm.intrinsics import compute_intrinsics, print_intrinsics

    image_paths = sorted(
        [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_path}")

    db = compute_intrinsics(image_paths)
    print(
        f"Found {db.num_images} images with EXIF data ({db.num_cameras} unique camera model(s)):\n"
    )
    print_intrinsics(db)

    db.save(output)
    print(f"\nSaved camera database → {output}")


if __name__ == "__main__":
    app()
