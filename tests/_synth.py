import cv2 as cv
import numpy as np


def rotation_y(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def project(K: np.ndarray, pose: np.ndarray, pts3D: np.ndarray) -> np.ndarray:
    """Project (N, 3) world points through a 3x4 pose + K to (N, 2) pixels."""
    Xh = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])
    x = (K @ pose @ Xh.T).T
    x = x / x[:, 2:3]
    return x[:, :2]


def make_keypoints(pts2d: np.ndarray):
    """Build a list of cv2.KeyPoint from an (N, 2) pixel-coordinate array."""
    return [cv.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in pts2d]


def make_matches(n: int):
    """Build a 1-to-1 list of cv2.DMatch (queryIdx==trainIdx==i)."""
    return [cv.DMatch(i, i, 0, 1.0) for i in range(n)]
