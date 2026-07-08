import contextlib
import io

import cv2 as cv
import numpy as np

from main import PerspectiveNPoint


def _make_pnp_scene(K, rng, R_true, t_true, n=12):
    """Generate 3D world points and their 2D projections in a camera with known pose."""
    pts3D = rng.uniform(-1.5, 1.5, (n, 3))
    pts3D[:, 2] += 6.0
    pose_true = np.hstack([R_true, t_true[:, None]])
    P_true = K @ pose_true
    Xh = np.hstack([pts3D, np.ones((n, 1))])
    x = (P_true @ Xh.T).T
    x = x / x[:, 2:3]
    return pts3D, x[:, :2], pose_true


def _reproj_error(K, pose, pts3D, pts2D):
    Xh = np.hstack([pts3D, np.ones((pts3D.shape[0], 1))])
    proj = (K @ pose @ Xh.T).T
    proj = proj / proj[:, 2:3]
    return np.abs(proj[:, :2] - pts2D).max()


def test_p3p_pure_translation_matches_opencv(K, rng):
    """P3P with identity rotation + pure translation must recover the pose (works today)."""
    R_true = np.eye(3)
    t_true = np.array([-1.0, 0.0, 0.0])
    pts3D, pts2D, pose_true = _make_pnp_scene(K, rng, R_true, t_true)

    pnp = PerspectiveNPoint([], K)
    with contextlib.redirect_stdout(io.StringIO()):
        pose = pnp._solve_p3p(pts3D[:4], pts2D[:4])

    assert pose is not None
    assert _reproj_error(K, pose, pts3D, pts2D) < 1.0
    assert np.linalg.norm(pose[:, :3] - R_true) < 1e-3
    assert np.linalg.norm(pose[:, 3] - t_true) < 1e-3


def test_p3p_matches_opencv_with_rotation(K, rng):
    """P3P with rotation must recover the same pose as cv2.solvePnP."""
    theta = np.radians(15)
    c, s = np.cos(theta), np.sin(theta)
    R_true = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    t_true = np.array([-1.0, 0.1, 0.0])
    pts3D, pts2D, pose_true = _make_pnp_scene(K, rng, R_true, t_true)

    pnp = PerspectiveNPoint([], K)
    with contextlib.redirect_stdout(io.StringIO()):
        pose = pnp._solve_p3p(pts3D[:4], pts2D[:4])

    assert pose is not None

    # Compare against OpenCV's P3P (3 pts) + 4th-pt disambiguation via solvePnP.
    ok, rvec, tvec = cv.solvePnP(
        pts3D[:4].astype(np.float64),
        pts2D[:4].astype(np.float64),
        K.astype(np.float64),
        None,
        flags=cv.SOLVEPNP_P3P,
    )
    assert ok
    R_cv, _ = cv.Rodrigues(rvec)
    t_cv = tvec.flatten()

    assert np.linalg.norm(pose[:, :3] - R_true) < 1e-3
    assert np.linalg.norm(pose[:, 3] - t_true) < 1e-3
    assert np.allclose(pose[:, :3], R_cv, atol=1e-3)
    assert _reproj_error(K, pose, pts3D, pts2D) < 1.0


def test_p3p_reprojection_error_pure_translation(K, rng):
    """Recovered pose must reproject all points (not just the 3+1 used) accurately."""
    R_true = np.eye(3)
    t_true = np.array([0.4, -0.3, 0.2])
    pts3D, pts2D, _ = _make_pnp_scene(K, rng, R_true, t_true, n=20)

    pnp = PerspectiveNPoint([], K)
    with contextlib.redirect_stdout(io.StringIO()):
        pose = pnp._solve_p3p(pts3D[:4], pts2D[:4])

    assert pose is not None
    assert _reproj_error(K, pose, pts3D, pts2D) < 0.5
