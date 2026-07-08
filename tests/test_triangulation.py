import cv2 as cv
import numpy as np
import pytest

from main import triangulate_pts_dlt


def test_triangulation_dlt_matches_ground_truth(stereo_scene):
    """On noise-free data the DLT triangulation must recover the exact 3D points."""
    s = stereo_scene
    pts = triangulate_pts_dlt(s["pts1"], s["pts2"], s["P1"], s["P2"])
    assert pts.shape == (s["pts3D"].shape[0], 3)
    assert np.allclose(pts, s["pts3D"], atol=1e-6)


def test_triangulation_dlt_matches_opencv(stereo_scene):
    """Our DLT triangulation must agree with cv2.triangulatePoints."""
    s = stereo_scene
    ours = triangulate_pts_dlt(s["pts1"], s["pts2"], s["P1"], s["P2"])

    cv_pts = cv.triangulatePoints(
        s["P1"].astype(np.float64),
        s["P2"].astype(np.float64),
        s["pts1"].T.astype(np.float64),
        s["pts2"].T.astype(np.float64),
    ).T
    cv_pts = cv_pts[:, :3] / cv_pts[:, 3:4]

    assert np.allclose(ours, cv_pts, atol=1e-6)


def test_triangulation_dlt_with_noise(stereo_scene, rng):
    """With ~1px Gaussian noise the triangulation stays close to ground truth."""
    s = stereo_scene
    pts1 = s["pts1"] + rng.normal(0, 1.0, s["pts1"].shape)
    pts2 = s["pts2"] + rng.normal(0, 1.0, s["pts2"].shape)

    pts = triangulate_pts_dlt(pts1, pts2, s["P1"], s["P2"])
    # 1px noise on a ~1000px focal length at depth ~8 should give small error.
    assert np.linalg.norm(pts - s["pts3D"], axis=1).mean() < 0.5


def test_cheirality_condition(stereo_scene):
    """Triangulated points must lie in front of both cameras (cheirality)."""
    s = stereo_scene
    pts = triangulate_pts_dlt(s["pts1"], s["pts2"], s["P1"], s["P2"])

    # Camera A at origin: depth is simply the Z coordinate.
    assert np.all(pts[:, 2] > 0)

    # Camera B: depth is r_3 @ X + t_b[2] (third row of [R|t] applied to X).
    R_b, t_b = s["R2"], s["t2"]
    depth_b = (R_b[2, :] @ pts.T) + t_b[2]
    assert np.all(depth_b > 0)


def test_triangulation_single_point_matches_opencv(stereo_scene):
    """The per-point linear system must match OpenCV for an isolated point."""
    s = stereo_scene
    i = 3
    xa = s["pts1"][i : i + 1]
    xb = s["pts2"][i : i + 1]
    ours = triangulate_pts_dlt(xa, xb, s["P1"], s["P2"])

    cv_pts = cv.triangulatePoints(
        s["P1"].astype(np.float64),
        s["P2"].astype(np.float64),
        xa.T.astype(np.float64),
        xb.T.astype(np.float64),
    ).T
    cv_pts = cv_pts[:, :3] / cv_pts[:, 3:4]

    assert np.allclose(ours, cv_pts, atol=1e-8)
    assert np.allclose(ours[0], s["pts3D"][i], atol=1e-6)
