import cv2 as cv
import numpy as np

from sfm.bootstrapping import StructureBootstrap
from sfm.epipolar_geometry import (
    EpipolarRANSAC,
    decompose_essential_matrix,
    select_pose_by_cheirality,
)


def _F(stereo_scene) -> np.ndarray:
    ransac = EpipolarRANSAC([])
    return ransac._compute_fundamental_matrix(stereo_scene["pts1"], stereo_scene["pts2"])


def _E(stereo_scene) -> np.ndarray:
    boot = StructureBootstrap([], stereo_scene["K"])
    return boot._compute_essential_matrix(_F(stereo_scene))


def _normalized_rays(K, pts):
    Kinv = np.linalg.inv(K)
    ph = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (Kinv @ ph.T).T


def test_essential_matrix_normalized_epipolar_constraint(stereo_scene):
    """E must satisfy y2^T E y1 = 0 for normalized image rays y = K^-1 x."""
    s = stereo_scene
    E = _E(s)
    y1 = _normalized_rays(s["K"], s["pts1"])
    y2 = _normalized_rays(s["K"], s["pts2"])
    res = np.abs((y2 * (y1 @ E.T)).sum(axis=1))
    assert res.max() < 1e-6


def test_essential_matrix_rank2(stereo_scene):
    """E must have rank 2."""
    E = _E(stereo_scene)
    assert np.linalg.matrix_rank(E, tol=1e-6) == 2


def test_essential_matrix_matches_opencv(stereo_scene):
    """Our E (= K^T F K) must match cv2.findEssentialMat up to scale/sign."""
    s = stereo_scene
    E_ours = _E(s)
    E_cv, _ = cv.findEssentialMat(
        s["pts1"].astype(np.float32),
        s["pts2"].astype(np.float32),
        s["K"].astype(np.float32),
        cv.FM_8POINT,
    )
    en_ours = E_ours / np.linalg.norm(E_ours)
    en_cv = E_cv / np.linalg.norm(E_cv)
    assert np.allclose(en_ours, en_cv, atol=1e-3) or np.allclose(
        en_ours, -en_cv, atol=1e-3
    )


def test_essential_decomposition_matches_opencv(stereo_scene):
    """decompose_essential_matrix must yield the same R1/R2/t as cv2.decomposeEssentialMat."""
    s = stereo_scene
    E = _E(s)
    R1_cv, R2_cv, t_cv = cv.decomposeEssentialMat(E)

    candidates = decompose_essential_matrix(E)
    R1 = candidates[0][:, :3]
    R2 = candidates[2][:, :3]
    t = candidates[0][:, 3]

    assert np.isclose(np.linalg.det(R1), 1.0)
    assert np.isclose(np.linalg.det(R2), 1.0)
    assert np.allclose(R1, R1_cv, atol=1e-6) or np.allclose(R1, R2_cv, atol=1e-6)
    assert np.allclose(R2, R2_cv, atol=1e-6) or np.allclose(R2, R1_cv, atol=1e-6)
    assert np.allclose(np.abs(t), np.abs(t_cv.flatten()), atol=1e-6)


def test_recover_pose_matches_opencv(stereo_scene):
    """select_pose_by_cheirality must agree with cv2.recoverPose on R and t-direction."""
    s = stereo_scene
    E = _E(s)
    _, R_cv, t_cv, _ = cv.recoverPose(
        E, s["pts1"].astype(np.float32), s["pts2"].astype(np.float32), s["K"].astype(np.float32)
    )

    candidates = decompose_essential_matrix(E)
    P1 = s["K"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best_pose, _ = select_pose_by_cheirality(
        candidates, s["pts1"], s["pts2"], P1, s["K"]
    )
    assert best_pose is not None
    R_best, t_best = best_pose[:, :3], best_pose[:, 3]

    assert np.allclose(R_best, R_cv, atol=1e-6)
    t_true_dir = s["t2"] / np.linalg.norm(s["t2"])
    assert np.allclose(np.abs(t_best @ t_true_dir), 1.0, atol=1e-3)


def test_decompose_essential_matrix_returns_four_valid_poses(stereo_scene):
    """All four candidates must be (3,4) [R|t] matrices with det(R)=1."""
    E = _E(stereo_scene)
    candidates = decompose_essential_matrix(E)
    assert len(candidates) == 4
    for pose in candidates:
        assert pose.shape == (3, 4)
        assert np.isclose(np.linalg.det(pose[:, :3]), 1.0)


def test_decompose_essential_matrix_two_distinct_rotations(stereo_scene):
    """Candidates 0,1 share R1; candidates 2,3 share R2; R1 != R2."""
    E = _E(stereo_scene)
    candidates = decompose_essential_matrix(E)
    R1a, R1b = candidates[0][:, :3], candidates[1][:, :3]
    R2a, R2b = candidates[2][:, :3], candidates[3][:, :3]
    assert np.allclose(R1a, R1b)
    assert np.allclose(R2a, R2b)
    assert not np.allclose(R1a, R2a)


def test_select_pose_by_cheirality_all_points_in_front(stereo_scene):
    """On noise-free data the selected pose must put every point in front of both cameras."""
    s = stereo_scene
    E = _E(s)
    candidates = decompose_essential_matrix(E)
    P1 = s["K"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best_pose, best_pts = select_pose_by_cheirality(
        candidates, s["pts1"], s["pts2"], P1, s["K"]
    )
    assert best_pose is not None
    assert best_pts is not None
    in_front_a = best_pts[:, 2] > 0
    in_front_b = (best_pts @ best_pose[2, :3] + best_pose[2, 3]) > 0
    assert in_front_a.all()
    assert in_front_b.all()


def test_select_pose_by_cheirality_returns_none_for_degenerate(stereo_scene):
    """With candidates that place no points in front, returns (None, None)."""
    s = stereo_scene
    # Camera looking 180° away (around Y) and placed behind the points so that
    # no triangulated point has positive depth in camera B.
    R_away = np.diag([-1.0, 1.0, -1.0])
    t_away = np.array([0.0, 0.0, -20.0])
    bad_pose = np.hstack([R_away, t_away[:, None]])
    bad_candidates = [bad_pose] * 4
    P1 = s["K"] @ np.hstack([np.eye(3), np.zeros((3, 1))])
    pts2_bad = (s["K"] @ bad_pose @ np.hstack([s["pts3D"], np.ones((s["pts3D"].shape[0], 1))]).T).T
    pts2_bad = pts2_bad[:, :2] / pts2_bad[:, 2:3]
    best_pose, best_pts = select_pose_by_cheirality(
        bad_candidates, s["pts1"], pts2_bad, P1, s["K"]
    )
    assert best_pose is None
    assert best_pts is None
