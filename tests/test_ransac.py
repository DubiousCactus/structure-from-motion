import contextlib
import io

import numpy as np
import pytest

from sfm.epipolar_geometry import (
    EpipolarRANSAC,
    eight_point_fundamental_matrix,
    sampson_distance,
)
from conftest import build_frame_tuple


def _run_filter(ransac: EpipolarRANSAC):
    """Run RANSAC.filter() with stdout (the print) and stderr (tqdm bars) swallowed."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return ransac.filter()


def test_ransac_recovers_inlier_mask(contaminated_scene):
    """filter() must return one boolean mask per frame tuple."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    assert len(inliers) == 1
    mask = inliers[0]
    assert mask.shape == (s["n_total"],)
    assert mask.dtype == bool


def test_ransac_finds_most_inliers(contaminated_scene):
    """RANSAC must recover (nearly) all ground-truth inliers."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    mask = inliers[0]
    recovered = int((mask & s["inlier_mask"]).sum())
    assert recovered >= int(0.9 * s["n_inliers"])


def test_ransac_rejects_outliers(contaminated_scene):
    """RANSAC must not classify many outliers as inliers (high precision)."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    mask = inliers[0]
    false_positives = int((mask & s["outlier_mask"]).sum())
    assert false_positives <= 0.1 * s["n_outliers"]


def test_ransac_stores_fundamental_matrix(contaminated_scene):
    """filter() must store the refined F on the FrameTuple."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    _run_filter(ransac)
    assert f_tpl.fundamental_matrix is not None
    assert f_tpl.fundamental_matrix.shape == (3, 3)
    assert np.linalg.matrix_rank(f_tpl.fundamental_matrix, tol=1e-8) == 2


def test_ransac_fundamental_matrix_satisfies_epipolar_constraint(contaminated_scene):
    """The stored F must satisfy the epipolar constraint for the true inliers."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    _run_filter(ransac)
    F = f_tpl.fundamental_matrix
    # The refined F is a least-squares fit over the RANSAC inlier set (which may
    # include a few false-positive outliers), so we check the epipolar constraint
    # on the known-true inliers with a tolerance that accommodates that.
    mask = s["inlier_mask"]
    p1h = np.hstack([s["pts1"][mask], np.ones((mask.sum(), 1))])
    p2h = np.hstack([s["pts2"][mask], np.ones((mask.sum(), 1))])
    res = np.abs((p2h * (p1h @ F.T)).sum(axis=1))
    assert res.max() < 2e-3


def test_ransac_matches_opencv_on_inliers(contaminated_scene):
    """The RANSAC-refined F must match cv2.findFundamentalMat on the inliers."""
    s = contaminated_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=500, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    mask = inliers[0]

    import cv2 as cv

    F_cv, _ = cv.findFundamentalMat(
        s["pts1"][mask].astype(np.float32),
        s["pts2"][mask].astype(np.float32),
        cv.FM_8POINT,
        3.0,
        0.99,
    )
    F_ours = f_tpl.fundamental_matrix
    fn_ours = F_ours / np.linalg.norm(F_ours)
    fn_cv = F_cv / np.linalg.norm(F_cv)
    assert np.allclose(fn_ours, fn_cv, atol=1e-3) or np.allclose(
        fn_ours, -fn_cv, atol=1e-3
    )


def test_ransac_pure_inliers_no_outliers(stereo_scene):
    """With zero outliers RANSAC must classify every point as an inlier."""
    s = stereo_scene
    f_tpl = build_frame_tuple(s["pts1"], s["pts2"])
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=200, consensus_ratio=0.5)
    inliers = _run_filter(ransac)
    assert inliers[0].all()


def test_ransac_skips_frame_with_too_few_matches(K, rng):
    """filter() must warn and skip frame tuples with fewer than 8 matches."""
    pts1 = rng.uniform([0, 0], [1000, 1000], (5, 2))
    pts2 = rng.uniform([0, 0], [1000, 1000], (5, 2))
    f_tpl = build_frame_tuple(pts1, pts2)
    ransac = EpipolarRANSAC([f_tpl], threshold=1.0, max_iter=10)
    with pytest.warns(UserWarning):
        inliers = _run_filter(ransac)
    # The frame is skipped: no inlier mask is appended.
    assert len(inliers) == 0
    assert f_tpl.fundamental_matrix is None


def test_ransac_threshold_controls_strictness(contaminated_scene):
    """A very small threshold must reject more points; a large one must accept more."""
    s = contaminated_scene
    f_tpl_strict = build_frame_tuple(s["pts1"], s["pts2"])
    f_tpl_loose = build_frame_tuple(s["pts1"], s["pts2"])

    ransac_strict = EpipolarRANSAC(
        [f_tpl_strict], threshold=0.01, max_iter=500, consensus_ratio=0.3
    )
    ransac_loose = EpipolarRANSAC(
        [f_tpl_loose], threshold=1e6, max_iter=500, consensus_ratio=0.3
    )
    strict = _run_filter(ransac_strict)[0]
    loose = _run_filter(ransac_loose)[0]
    # The loose threshold admits every point (inliers + outliers).
    assert loose.sum() == s["n_total"]
    # The strict threshold admits fewer than the loose one.
    assert strict.sum() < loose.sum()


def test_ransac_multiple_frame_tuples(contaminated_scene, stereo_scene):
    """filter() must process multiple frame tuples and return one mask each."""
    f_a = build_frame_tuple(contaminated_scene["pts1"], contaminated_scene["pts2"])
    f_b = build_frame_tuple(stereo_scene["pts1"], stereo_scene["pts2"])
    ransac = EpipolarRANSAC([f_a, f_b], threshold=1.0, max_iter=300, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    assert len(inliers) == 2
    assert inliers[0].shape == (contaminated_scene["n_total"],)
    assert inliers[1].shape == (stereo_scene["pts1"].shape[0],)
    # Both frame tuples must have a fundamental matrix stored.
    assert f_a.fundamental_matrix is not None
    assert f_b.fundamental_matrix is not None


def test_ransac_refined_f_better_than_minimal_f(contaminated_scene, K, rng):
    """With noisy inliers, the F re-fit on all inliers must beat a minimal 8-pt F."""
    s = contaminated_scene
    # Add pixel noise to the inliers so a minimal 8-point sample is noisy.
    pts1 = s["pts1"].copy()
    pts2 = s["pts2"].copy()
    pts1[s["inlier_mask"]] += rng.normal(0, 1.0, (s["n_inliers"], 2))
    pts2[s["inlier_mask"]] += rng.normal(0, 1.0, (s["n_inliers"], 2))
    f_tpl = build_frame_tuple(pts1, pts2)
    ransac = EpipolarRANSAC([f_tpl], threshold=2.0, max_iter=500, consensus_ratio=0.3)
    inliers = _run_filter(ransac)
    mask = inliers[0]
    F_refined = f_tpl.fundamental_matrix

    # Fit an F on just the first 8 RANSAC inliers (minimal sample).
    inlier_idx = np.where(mask)[0][:8]
    F_minimal = eight_point_fundamental_matrix(pts1[inlier_idx], pts2[inlier_idx])

    true_inliers = s["inlier_mask"]
    err_refined = sampson_distance(F_refined, pts1[true_inliers], pts2[true_inliers]).mean()
    err_minimal = sampson_distance(F_minimal, pts1[true_inliers], pts2[true_inliers]).mean()
    assert err_refined < err_minimal
