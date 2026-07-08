import numpy as np
import pytest

from sfm.data import FrameTuple, ImageFeatures
from _synth import make_keypoints, make_matches, project, rotation_y


@pytest.fixture
def K() -> np.ndarray:
    return np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def stereo_scene(K, rng):
    """A noise-free two-view scene with known poses and 3D points.

    Returns a dict with poses, projection matrices, 3D points and the
    corresponding 2D pixel observations in both views.
    """
    R1 = np.eye(3)
    t1 = np.zeros(3)
    pose1 = np.hstack([R1, t1[:, None]])

    R2 = rotation_y(np.radians(15))
    t2 = np.array([-1.0, 0.2, 0.1])
    pose2 = np.hstack([R2, t2[:, None]])

    P1 = K @ pose1
    P2 = K @ pose2

    n = 80
    pts3D = rng.uniform(-2.0, 2.0, (n, 3))
    pts3D[:, 2] += 8.0  # depth ~6-10, well in front of both cameras

    pts1 = project(K, pose1, pts3D)
    pts2 = project(K, pose2, pts3D)

    return {
        "K": K,
        "R1": R1,
        "t1": t1,
        "pose1": pose1,
        "P1": P1,
        "R2": R2,
        "t2": t2,
        "pose2": pose2,
        "P2": P2,
        "pts3D": pts3D,
        "pts1": pts1,
        "pts2": pts2,
    }


@pytest.fixture
def contaminated_scene(K, rng):
    """A two-view scene with known inliers plus random outlier correspondences.

    The inliers are exact projections of 3D points through two known cameras.
    The outliers are random, independent 2D points in each view that do not
    satisfy any shared epipolar geometry. The ground-truth inlier mask is
    provided so tests can measure RANSAC precision/recall.

    Returns a dict with everything in ``stereo_scene`` plus ``inlier_mask``,
    ``outlier_mask``, and the contaminated ``pts1``/``pts2`` arrays.
    """
    R1 = np.eye(3)
    t1 = np.zeros(3)
    pose1 = np.hstack([R1, t1[:, None]])

    R2 = rotation_y(np.radians(15))
    t2 = np.array([-1.0, 0.2, 0.1])
    pose2 = np.hstack([R2, t2[:, None]])

    P1 = K @ pose1
    P2 = K @ pose2

    n_inliers = 60
    n_outliers = 40
    n_total = n_inliers + n_outliers

    pts3D = rng.uniform(-2.0, 2.0, (n_inliers, 3))
    pts3D[:, 2] += 8.0

    inlier_pts1 = project(K, pose1, pts3D)
    inlier_pts2 = project(K, pose2, pts3D)

    outlier_pts1 = rng.uniform([0, 0], [1000, 1000], (n_outliers, 2))
    outlier_pts2 = rng.uniform([0, 0], [1000, 1000], (n_outliers, 2))

    pts1 = np.vstack([inlier_pts1, outlier_pts1])
    pts2 = np.vstack([inlier_pts2, outlier_pts2])
    inlier_mask = np.concatenate([np.ones(n_inliers, bool), np.zeros(n_outliers, bool)])
    outlier_mask = ~inlier_mask

    return {
        "K": K,
        "R1": R1,
        "t1": t1,
        "pose1": pose1,
        "P1": P1,
        "R2": R2,
        "t2": t2,
        "pose2": pose2,
        "P2": P2,
        "pts3D": pts3D,
        "pts1": pts1,
        "pts2": pts2,
        "inlier_mask": inlier_mask,
        "outlier_mask": outlier_mask,
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "n_total": n_total,
    }


@pytest.fixture
def pnp_scene(K, rng):
    """A two-view scene with known inlier/outlier 3D-2D correspondences for PnP-RANSAC.

    Camera A is at the origin (identity).  Camera B has a known rotation +
    translation.  Some correspondences are outliers (randomly placed 2D
    observations in camera B that do *not* match the true projection).
    """
    R_a = np.eye(3)
    t_a = np.zeros(3)
    pose_a = np.hstack([R_a, t_a[:, None]])

    R_b = rotation_y(np.radians(15))
    t_b = np.array([-1.0, 0.2, 0.1])
    pose_b = np.hstack([R_b, t_b[:, None]])

    n_inliers = 40
    n_outliers = 20
    n_total = n_inliers + n_outliers

    pts3D = rng.uniform(-2.0, 2.0, (n_total, 3))
    pts3D[:, 2] += 8.0

    pts2D_a = project(K, pose_a, pts3D)
    pts2D_b = project(K, pose_b, pts3D)

    # Overwrite last n_outliers with random 2D points (outliers)
    pts2D_b[n_inliers:] = rng.uniform([0, 0], [1000, 1000], (n_outliers, 2))

    inlier_mask = np.zeros(n_total, dtype=bool)
    inlier_mask[:n_inliers] = True

    return {
        "K": K,
        "R_a": R_a,
        "t_a": t_a,
        "pose_a": pose_a,
        "R_b": R_b,
        "t_b": t_b,
        "pose_b": pose_b,
        "pts3D": pts3D,
        "pts2D_a": pts2D_a,
        "pts2D_b": pts2D_b,
        "inlier_mask": inlier_mask,
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "n_total": n_total,
    }


def build_frame_tuple(
    pts1: np.ndarray, pts2: np.ndarray, frame_a_id=0, frame_b_id=1
) -> FrameTuple:
    """Build a 1-to-1 FrameTuple from two (N, 2) pixel-coordinate arrays."""
    feats_a = ImageFeatures(make_keypoints(pts1), None, "frame_a")
    feats_b = ImageFeatures(make_keypoints(pts2), None, "frame_b")
    matches = make_matches(pts1.shape[0])
    return FrameTuple(frame_a_id, frame_b_id, feats_a, feats_b, matches)
