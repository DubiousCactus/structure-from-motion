from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import ExifTags, Image

from sfm.data import CameraDatabase


def read_exif(image_path: str) -> dict:
    img: Image.Image = Image.open(image_path)
    exif_data = img._getexif()
    if exif_data is None:
        raise ValueError(f"No EXIF data found in {image_path}")
    return {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}


def exif_to_K(tag_dict: dict, img_width: int, img_height: int) -> np.ndarray:
    focal_mm: Optional[float] = tag_dict.get("FocalLength")
    focal_35mm: Optional[float] = tag_dict.get("FocalLengthIn35mmFilm")
    fp_xres: Optional[float] = tag_dict.get("FocalPlaneXResolution")
    fp_yres: Optional[float] = tag_dict.get("FocalPlaneYResolution")
    fp_unit: Optional[int] = tag_dict.get("FocalPlaneResolutionUnit")

    cx = img_width / 2.0
    cy = img_height / 2.0

    if focal_35mm is not None:
        fx = focal_35mm * img_width / 36.0
        fy = focal_35mm * img_height / 24.0
    else:
        if focal_mm is None:
            raise ValueError("FocalLength tag not found in EXIF")
        if fp_xres is not None and fp_yres is not None and fp_unit is not None:
            if fp_unit == 2:  # inches
                px_per_mm_x = fp_xres / 25.4
                px_per_mm_y = fp_yres / 25.4
            elif fp_unit == 3:  # cm
                px_per_mm_x = fp_xres / 10.0
                px_per_mm_y = fp_yres / 10.0
            elif fp_unit == 4:  # mm
                px_per_mm_x = fp_xres
                px_per_mm_y = fp_yres
            else:
                raise ValueError(f"Unknown FocalPlaneResolutionUnit: {fp_unit}")
            fx = focal_mm * px_per_mm_x
            fy = focal_mm * px_per_mm_y
        else:
            raise ValueError(
                "Cannot compute focal length in pixels: "
                "need FocalPlaneResolution or FocalLengthIn35mmFilm in EXIF"
            )

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K


def _dedup_key(K: np.ndarray, width: int, height: int) -> bytes:
    return K.tobytes() + width.to_bytes(4, "little") + height.to_bytes(4, "little")


def compute_intrinsics(image_paths: list[str]) -> CameraDatabase:
    unique: dict[bytes, int] = {}
    cameras: list[np.ndarray] = []
    camera_ids: dict[str, int] = {}
    image_sizes: dict[str, tuple[int, int]] = {}

    for p in image_paths:
        basename = os.path.basename(p)
        img = Image.open(p)
        w, h = img.size
        exif = read_exif(p)
        K = exif_to_K(exif, w, h)
        key = _dedup_key(K, w, h)
        if key not in unique:
            unique[key] = len(cameras)
            cameras.append(K)
        camera_ids[basename] = unique[key]
        image_sizes[basename] = (w, h)

    return CameraDatabase(cameras, camera_ids, image_sizes)


def print_intrinsics(db: CameraDatabase) -> None:
    for basename, cid in sorted(db.camera_ids.items()):
        K = db.cameras[cid]
        w, h = db.image_sizes[basename]
        K_str = f"[{K[0, 0]:.1f} 0 {K[0, 2]:.1f}; 0 {K[1, 1]:.1f} {K[1, 2]:.1f}; 0 0 1]"
        print(f"  {basename} ({w}x{h})")
        print(f"    K = {K_str}")

    for i, K in enumerate(db.cameras):
        K_str = f"[{K[0, 0]:.1f} 0 {K[0, 2]:.1f}; 0 {K[1, 1]:.1f} {K[1, 2]:.1f}; 0 0 1]"
        print(f"\nCamera {i}: {K_str}")
    K_avg = db.average_K
    K_avg_str = f"[{K_avg[0, 0]:.1f} 0 {K_avg[0, 2]:.1f}; 0 {K_avg[1, 1]:.1f} {K_avg[1, 2]:.1f}; 0 0 1]"
    print(f"\nAverage K = {K_avg_str}")
    print(f"\n{len(db.cameras)} unique camera model(s), {db.num_images} image(s)")
