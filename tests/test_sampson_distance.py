import cv2 as cv
import numpy as np

from main import eight_point_fundamental_matrix, sampson_distance


def test_sampson_distance_zero_for_exact_correspondences(stereo_scene):
    """Sampson distance must be ~0 for noise-free correspondences."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    d = sampson_distance(F, s["pts1"], s["pts2"])
    assert d.shape == (s["pts1"].shape[0],)
    assert np.all(d < 1e-10)


def test_sampson_distance_non_negative(stereo_scene, rng):
    """Sampson distance is a squared distance and must be non-negative."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    pts2_noisy = s["pts2"] + rng.normal(0, 5.0, s["pts2"].shape)
    d = sampson_distance(F, s["pts1"], pts2_noisy)
    assert np.all(d >= 0.0)


def test_sampson_distance_increases_with_noise(stereo_scene, rng):
    """Larger pixel perturbations must yield larger Sampson distances on average."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    small = sampson_distance(F, s["pts1"], s["pts2"] + rng.normal(0, 0.5, s["pts2"].shape))
    large = sampson_distance(F, s["pts1"], s["pts2"] + rng.normal(0, 10.0, s["pts2"].shape))
    assert large.mean() > small.mean()


def test_sampson_distance_outliers_higher_than_inliers(stereo_scene, rng):
    """Random outlier matches must have larger Sampson distance than inliers."""
    s = stereo_scene
    n = s["pts1"].shape[0]
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    inlier_d = sampson_distance(F, s["pts1"], s["pts2"])

    outliers1 = rng.uniform([0, 0], [1000, 1000], (n, 2))
    outliers2 = rng.uniform([0, 0], [1000, 1000], (n, 2))
    outlier_d = sampson_distance(F, outliers1, outliers2)

    assert outlier_d.mean() > inlier_d.mean()


def test_sampson_distance_matches_manual_formula(stereo_scene, rng):
    """sampson_distance must match the reference formula computed inline."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    pts2_noisy = s["pts2"] + rng.normal(0, 3.0, s["pts2"].shape)

    d = sampson_distance(F, s["pts1"], pts2_noisy)

    # Reference inline computation.
    n = s["pts1"].shape[0]
    p1h = np.hstack([s["pts1"], np.ones((n, 1))])
    p2h = np.hstack([pts2_noisy, np.ones((n, 1))])
    Fx1 = p1h @ F.T
    Ftx2 = p2h @ F
    numer = (p2h * Fx1).sum(axis=1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    expected = numer / np.maximum(denom, 1e-12)

    assert np.allclose(d, expected)


def test_sampson_distance_threshold_separates_inliers_outliers(stereo_scene, rng):
    """A reasonable Sampson threshold must separate inliers from random outliers."""
    s = stereo_scene
    n_in = 60
    n_out = 40
    F = eight_point_fundamental_matrix(s["pts1"][:n_in], s["pts2"][:n_in])

    outliers1 = rng.uniform([0, 0], [1000, 1000], (n_out, 2))
    outliers2 = rng.uniform([0, 0], [1000, 1000], (n_out, 2))
    inlier_d = sampson_distance(F, s["pts1"][:n_in], s["pts2"][:n_in])
    outlier_d = sampson_distance(F, outliers1, outliers2)

    threshold = 1.0
    assert (inlier_d < threshold).mean() > 0.95
    assert (outlier_d > threshold).mean() > 0.95
