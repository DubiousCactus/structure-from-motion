import contextlib
import io

import numpy as np

from sfm.bootstrapping import StructureBootstrap
from sfm.epipolar_geometry import EpipolarRANSAC
from conftest import build_frame_tuple


def _build_frame_tuple(stereo_scene):
    s = stereo_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])

    # Set the fundamental matrix using our own normalized 8-point solver so the
    # whole F -> E -> pose -> triangulation pipeline is exercised.
    ransac = EpipolarRANSAC([])
    f_tpl.fundamental_matrix = ransac._compute_fundamental_matrix(s["pts1"], s["pts2"])
    return f_tpl


def test_bootstrap_recovers_rotation(stereo_scene):
    """StructureBootstrap.init must recover camera B's rotation exactly."""
    s = stereo_scene
    f_tpl = _build_frame_tuple(s)
    boot = StructureBootstrap([f_tpl], s["K"])
    inliers = np.ones(s["pts3D"].shape[0], dtype=bool)

    with contextlib.redirect_stdout(io.StringIO()):
        structure = boot.init([inliers])

    R_b = structure.poses[1][:, :3]
    assert np.allclose(R_b, s["R2"], atol=1e-4)
    assert np.isclose(np.linalg.det(R_b), 1.0)


def test_bootstrap_recovers_translation_direction(stereo_scene):
    """E-decomposition only fixes translation up to scale; its direction must match."""
    s = stereo_scene
    f_tpl = _build_frame_tuple(s)
    boot = StructureBootstrap([f_tpl], s["K"])
    inliers = np.ones(s["pts3D"].shape[0], dtype=bool)

    with contextlib.redirect_stdout(io.StringIO()):
        structure = boot.init([inliers])

    t_b = structure.poses[1][:, 3]
    t_true_dir = s["t2"] / np.linalg.norm(s["t2"])
    # Direction must match (up to the sign chosen by cheirality, which is the
    # physically correct one here).
    assert np.allclose(t_b / np.linalg.norm(t_b), t_true_dir, atol=1e-3)


def test_bootstrap_pose_a_is_identity(stereo_scene):
    """Camera A must be anchored at the world origin [I|0]."""
    s = stereo_scene
    f_tpl = _build_frame_tuple(s)
    boot = StructureBootstrap([f_tpl], s["K"])
    inliers = np.ones(s["pts3D"].shape[0], dtype=bool)

    with contextlib.redirect_stdout(io.StringIO()):
        structure = boot.init([inliers])

    pose_a = structure.poses[0]
    assert np.allclose(pose_a, np.hstack([np.eye(3), np.zeros((3, 1))]))


def test_bootstrap_points_match_ground_truth_up_to_scale(stereo_scene):
    """Triangulated points must match the ground-truth cloud up to a single global scale."""
    s = stereo_scene
    f_tpl = _build_frame_tuple(s)
    boot = StructureBootstrap([f_tpl], s["K"])
    inliers = np.ones(s["pts3D"].shape[0], dtype=bool)

    with contextlib.redirect_stdout(io.StringIO()):
        structure = boot.init([inliers])

    rec = structure.points3D
    gt = s["pts3D"]
    # Least-squares global scale: minimize ||s*rec - gt||^2  =>  s = <rec,gt>/<rec,rec>.
    scale = float(rec.ravel() @ gt.ravel()) / float(rec.ravel() @ rec.ravel())
    err = np.linalg.norm(scale * rec - gt, axis=1)
    assert err.mean() < 1e-2


def test_bootstrap_refinement_improves_or_matches_linear(stereo_scene):
    """Non-linear triangulation should not be worse than the linear estimate."""
    s = stereo_scene
    f_tpl = _build_frame_tuple(s)
    boot = StructureBootstrap([f_tpl], s["K"])
    inliers = np.ones(s["pts3D"].shape[0], dtype=bool)

    with contextlib.redirect_stdout(io.StringIO()):
        structure = boot.init([inliers])

    gt = s["pts3D"]
    lin = f_tpl.triangulated_pts_linear
    ref = structure.points3D

    def scaled_err(pts):
        sc = float(pts.ravel() @ gt.ravel()) / float(pts.ravel() @ pts.ravel())
        return np.linalg.norm(sc * pts - gt, axis=1).mean()

    assert scaled_err(ref) <= scaled_err(lin) + 1e-6
