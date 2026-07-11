import cv2 as cv
import numpy as np

from sfm.data import CameraDatabase
from sfm.pnp import PerspectiveNPoint


def _reproj_error(K, pose, pts3D, pts2D):
    """Max reprojection error in pixels."""
    Xh = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])
    proj = (K @ pose @ Xh.T).T
    proj = proj / proj[:, 2:3]
    return np.abs(proj[:, :2] - pts2D).max()


def test_pnp_ransac_recovers_pose(pnp_scene):
    """RANSAC-P3P must recover the ground-truth pose of camera B from mixed inliers
    and outliers."""
    s = pnp_scene
    cam_db = CameraDatabase.from_single(s["K"])
    pnp = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=0.01, ransac_iter=2000, consensus_ratio=0.3
    )
    consensus_min = max(4, int(0.3 * s["n_total"]))

    pose, pts3D = pnp._ransac_pnp(
        s["pts3D"],
        s["pts2D_a"],
        s["pts2D_b"],
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )

    assert pose is not None
    assert (
        _reproj_error(s["K"], pose, s["pts3D"][s["inlier_mask"]], s["pts2D_b"][s["inlier_mask"]])
        < 1.0
    )
    assert np.linalg.norm(pose[:, :3] - s["R_b"]) < 1e-3
    assert np.linalg.norm(pose[:, 3] - s["t_b"]) < 1e-3


def test_pnp_ransac_pure_inliers(pnp_scene):
    """With all inliers, RANSAC-P3P must recover the exact pose."""
    s = pnp_scene
    # Use only the inlier subset
    mask = s["inlier_mask"]
    cam_db = CameraDatabase.from_single(s["K"])
    pnp = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=0.01, ransac_iter=500, consensus_ratio=0.3
    )
    consensus_min = max(4, int(0.3 * mask.sum()))

    pose, pts3D = pnp._ransac_pnp(
        s["pts3D"][mask],
        s["pts2D_a"][mask],
        s["pts2D_b"][mask],
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )

    assert pose is not None
    assert _reproj_error(s["K"], pose, s["pts3D"][mask], s["pts2D_b"][mask]) < 0.5
    assert np.linalg.norm(pose[:, :3] - s["R_b"]) < 1e-4
    assert np.linalg.norm(pose[:, 3] - s["t_b"]) < 1e-4


def test_pnp_ransac_matches_opencv(pnp_scene):
    """RANSAC-P3P pose must be close to cv2.solvePnPRansac."""
    s = pnp_scene
    cam_db = CameraDatabase.from_single(s["K"])
    pnp = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=0.01, ransac_iter=2000, consensus_ratio=0.3
    )
    consensus_min = max(4, int(0.3 * s["n_total"]))

    pose, pts3D = pnp._ransac_pnp(
        s["pts3D"],
        s["pts2D_a"],
        s["pts2D_b"],
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )
    assert pose is not None

    ok, rvec, tvec, inliers_cv = cv.solvePnPRansac(
        s["pts3D"].astype(np.float64),
        s["pts2D_b"].astype(np.float64),
        s["K"].astype(np.float64),
        None,
        confidence=0.99,
        reprojectionError=1.0,
        iterationsCount=2000,
        flags=cv.SOLVEPNP_P3P,
    )
    assert ok

    R_cv, _ = cv.Rodrigues(rvec)
    t_cv = tvec.flatten()

    assert np.linalg.norm(pose[:, :3] - R_cv) < 1e-3
    assert np.linalg.norm(pose[:, 3] - t_cv) < 5e-2


def test_pnp_ransac_threshold_controls_strictness(pnp_scene, rng):
    """With noisy inliers, a tight threshold rejects more points than a generous one."""
    s = pnp_scene
    # Add pixel noise so exact reprojection is impossible.  With stddev 5.0
    # the expected squared error ~25, so a threshold of 1e-8 rejects everything
    # while a threshold of 1e6 accepts everyone.
    noisy_pts2D_b = s["pts2D_b"] + rng.normal(0, 5.0, s["pts2D_b"].shape)

    cam_db = CameraDatabase.from_single(s["K"])
    consensus_min = max(4, int(0.3 * s["n_total"]))

    # Tiny threshold — no point can satisfy it with noise stddev 5.
    pnp_strict = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=1e-8, ransac_iter=2000
    )
    pose_strict, _ = pnp_strict._ransac_pnp(
        s["pts3D"],
        s["pts2D_a"],
        noisy_pts2D_b,
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )
    assert pose_strict is None

    # Generous threshold — every point passes.
    pnp_loose = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=1e6, ransac_iter=500
    )
    pose_loose, _ = pnp_loose._ransac_pnp(
        s["pts3D"],
        s["pts2D_a"],
        noisy_pts2D_b,
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )
    assert pose_loose is not None


def test_pnp_ransac_insufficient_inliers(pnp_scene):
    """When consensus_min > total correspondences, _ransac_pnp returns (None, None)."""
    s = pnp_scene
    cam_db = CameraDatabase.from_single(s["K"])
    pnp = PerspectiveNPoint(
        [], cam_db, ransac_inlier_threshold=0.01, ransac_iter=100, consensus_ratio=0.99
    )
    consensus_min = 10_000  # far more than available points

    pose, pts3D = pnp._ransac_pnp(
        s["pts3D"],
        s["pts2D_a"],
        s["pts2D_b"],
        s["K"],
        s["K"],
        s["pose_a"],
        consensus_min,
    )

    assert pose is None
    assert pts3D is None
