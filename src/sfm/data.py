import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


@dataclass
class ImageFeatures:
    keypoints: List[cv.KeyPoint]
    descriptors: List[np.ndarray]
    img_path: str


@dataclass
class FrameTuple:
    frame_a_id: int
    frame_b_id: int
    frame_a_features: ImageFeatures
    frame_b_features: ImageFeatures
    frame_a_to_b_matches: Sequence[cv.DMatch]
    fundamental_matrix: Optional[np.ndarray] = None
    essential_matrix: Optional[np.ndarray] = None
    cam_pose_a: Optional[np.ndarray] = None
    cam_pose_b: Optional[np.ndarray] = None
    inliers: Optional[int] = 0
    triangulated_pts_linear: Optional[np.ndarray] = None
    triangulated_pts: Optional[np.ndarray] = None


@dataclass
class Structure:
    points3D: np.ndarray
    correspondences: Dict[Tuple[int, int], int]
    poses: Dict[int, np.ndarray]


@dataclass
class CameraDatabase:
    cameras: List[np.ndarray] = field(default_factory=list)
    camera_ids: Dict[str, int] = field(default_factory=dict)
    image_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    default_K: Optional[np.ndarray] = None

    def get_K(self, img_path: str) -> np.ndarray:
        basename = os.path.basename(img_path)
        cid = self.camera_ids.get(basename)
        if cid is not None:
            return self.cameras[cid]
        if self.default_K is not None:
            return self.default_K
        raise KeyError(f"No intrinsics for {basename}")

    def get_size(self, img_path: str) -> Tuple[int, int]:
        return self.image_sizes[os.path.basename(img_path)]

    @property
    def average_K(self) -> np.ndarray:
        if len(self.cameras) == 1:
            return self.cameras[0]
        return np.mean(self.cameras, axis=0)

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_images(self) -> int:
        return len(self.camera_ids)

    def save(self, path: str) -> None:
        basenames = sorted(self.camera_ids)
        camera_ids_arr = np.array(
            [self.camera_ids[b] for b in basenames], dtype=np.int32
        )
        widths_arr = np.array(
            [self.image_sizes[b][0] for b in basenames], dtype=np.int32
        )
        heights_arr = np.array(
            [self.image_sizes[b][1] for b in basenames], dtype=np.int32
        )
        np.savez_compressed(
            path,
            cameras=np.stack(self.cameras),
            basenames=np.array(basenames),
            camera_ids=camera_ids_arr,
            widths=widths_arr,
            heights=heights_arr,
            K_avg=self.average_K,
        )

    @staticmethod
    def load(path: str) -> "CameraDatabase":
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            cs = data["cameras"]
            cameras = [cs[i] for i in range(cs.shape[0])]
            basenames = data["basenames"].tolist()
            camera_ids = {b: int(cid) for b, cid in zip(basenames, data["camera_ids"])}
            sizes = {
                b: (int(w), int(h))
                for b, w, h in zip(basenames, data["widths"], data["heights"])
            }
            return CameraDatabase(cameras, camera_ids, sizes)
        arr = np.asarray(data)
        if arr.ndim == 3:
            K = arr[0]
        else:
            K = arr
        return CameraDatabase([K], {}, {}, default_K=K)

    @staticmethod
    def from_single(
        K: np.ndarray,
        basenames: Optional[List[str]] = None,
    ) -> "CameraDatabase":
        basenames = basenames or []
        sizes = {}
        return CameraDatabase(
            cameras=[K],
            camera_ids={b: 0 for b in basenames},
            image_sizes=sizes,
            default_K=K,
        )
