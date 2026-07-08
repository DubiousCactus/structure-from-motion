import cv2 as cv
import numpy as np
import pytest

from main import eight_point_fundamental_matrix, hartley_normalize


def test_hartley_normalize_centroid_at_origin(rng):
    """After normalization the centroid of the points must be at the origin."""
    pts = rng.uniform([0, 0], [1000, 1000], (50, 2))
    normed, _ = hartley_normalize(pts)
    assert np.allclose(normed.mean(axis=0), 0.0, atol=1e-10)


def test_hartley_normalize_mean_distance_sqrt2(rng):
    """After normalization the mean distance from the origin must be sqrt(2)."""
    pts = rng.uniform([0, 0], [1000, 1000], (50, 2))
    normed, _ = hartley_normalize(pts)
    mean_dist = np.linalg.norm(normed, axis=1).mean()
    assert np.isclose(mean_dist, np.sqrt(2), atol=1e-10)


def test_hartley_normalize_transform_shape_and_type(rng):
    """The returned transform T must be a (3, 3) array."""
    pts = rng.uniform([0, 0], [1000, 1000], (20, 2))
    _, T = hartley_normalize(pts)
    assert T.shape == (3, 3)
    assert T.dtype == pts.dtype


def test_hartley_normalize_transform_applies_to_points(rng):
    """T must map the homogeneous input points to the normalized points."""
    pts = rng.uniform([0, 0], [1000, 1000], (30, 2))
    normed, T = hartley_normalize(pts)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    reconstructed = (T @ pts_h.T).T[:, :2]
    assert np.allclose(reconstructed, normed, atol=1e-10)


def test_hartley_normalize_preserves_shape(rng):
    """The normalized points must have the same shape as the input."""
    pts = rng.uniform([0, 0], [1000, 1000], (40, 2))
    normed, _ = hartley_normalize(pts)
    assert normed.shape == pts.shape


def test_hartley_normalize_invariance_to_translation(rng):
    """Translating all input points by a constant must not change the normalized output."""
    pts = rng.uniform([0, 0], [10, 10], (50, 2))
    offset = np.array([500.0, 700.0])
    normed_a, _ = hartley_normalize(pts)
    normed_b, _ = hartley_normalize(pts + offset)
    assert np.allclose(normed_a, normed_b, atol=1e-10)


def test_hartley_normalize_invariance_to_uniform_scale(rng):
    """Uniformly scaling the input must not change the normalized output."""
    pts = rng.uniform([0, 0], [10, 10], (50, 2))
    normed_a, _ = hartley_normalize(pts)
    normed_b, _ = hartley_normalize(pts * 100.0)
    assert np.allclose(normed_a, normed_b, atol=1e-10)


def test_hartley_normalize_rejects_wrong_shape(rng):
    """Must reject inputs that are not (N, 2)."""
    with pytest.raises(AssertionError):
        hartley_normalize(rng.uniform(0, 1, (10, 3)))
    with pytest.raises(AssertionError):
        hartley_normalize(rng.uniform(0, 1, 10))


def test_eight_point_matches_opencv(stereo_scene):
    """eight_point_fundamental_matrix must match cv2 (FM_8POINT) up to scale/sign."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    F_cv, _ = cv.findFundamentalMat(
        s["pts1"].astype(np.float32),
        s["pts2"].astype(np.float32),
        cv.FM_8POINT,
        3.0,
        0.99,
    )
    fn = F / np.linalg.norm(F)
    fn_cv = F_cv / np.linalg.norm(F_cv)
    assert np.allclose(fn, fn_cv, atol=1e-3) or np.allclose(fn, -fn_cv, atol=1e-3)


def test_eight_point_epipolar_constraint(stereo_scene):
    """All correspondences must satisfy x2^T F x1 = 0."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    p1h = np.hstack([s["pts1"], np.ones((s["pts1"].shape[0], 1))])
    p2h = np.hstack([s["pts2"], np.ones((s["pts2"].shape[0], 1))])
    res = np.abs((p2h * (p1h @ F.T)).sum(axis=1))
    assert res.max() < 1e-6


def test_eight_point_rank2(stereo_scene):
    """F must have rank 2 (the rank-2 constraint is enforced)."""
    s = stereo_scene
    F = eight_point_fundamental_matrix(s["pts1"], s["pts2"])
    assert np.linalg.matrix_rank(F, tol=1e-8) == 2


def test_eight_point_requires_eight_points(stereo_scene):
    """Must reject fewer than 8 correspondences."""
    s = stereo_scene
    with pytest.raises(AssertionError):
        eight_point_fundamental_matrix(s["pts1"][:7], s["pts2"][:7])


def test_eight_point_rejects_mismatched_counts(stereo_scene):
    """Must reject mismatched correspondence counts."""
    s = stereo_scene
    with pytest.raises(AssertionError):
        eight_point_fundamental_matrix(s["pts1"][:10], s["pts2"][:12])


def test_eight_point_raises_on_degenerate_coplanar(rng):
    """Coplanar (collinear in the design-matrix sense) points must raise ValueError."""
    # All points on a single line -> design matrix rank-deficient.
    t = rng.uniform(0, 1, 12)
    pts1 = np.column_stack([t, 2 * t + 1])
    pts2 = np.column_stack([t, 3 * t + 5])
    with pytest.raises(ValueError):
        eight_point_fundamental_matrix(pts1, pts2)


def test_hartley_normalize_translation_invariant(stereo_scene):
    """Hartley normalization produces the same normalized points irrespective of
    an additive translation of the input."""
    s = stereo_scene
    offset = np.array([200.0, 150.0])
    n_a, _ = hartley_normalize(s["pts1"])
    n_b, _ = hartley_normalize(s["pts1"] + offset)
    assert np.allclose(n_a, n_b, atol=1e-10)


def test_eight_point_shifted_input_satisfies_epipolar_constraint(stereo_scene):
    """F computed from translated pixel coords must satisfy the epipolar
    constraint on the translated points."""
    s = stereo_scene
    offset = np.array([200.0, 150.0])
    pts1_s = s["pts1"] + offset
    pts2_s = s["pts2"] + offset
    F = eight_point_fundamental_matrix(pts1_s, pts2_s)
    p1h = np.hstack([pts1_s, np.ones((pts1_s.shape[0], 1))])
    p2h = np.hstack([pts2_s, np.ones((pts2_s.shape[0], 1))])
    assert np.abs((p2h * (p1h @ F.T)).sum(axis=1)).max() < 1e-6
