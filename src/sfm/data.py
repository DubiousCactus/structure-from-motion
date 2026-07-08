import cv2 as cv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
