import numpy as np

from sfm.feature_extraction import match_orb_descriptors


def test_match_orb_descriptors_requires_reciprocal_lowe_matches():
    descriptors_a = np.array([[0, 0], [0, 255], [0, 1]], dtype=np.uint8)
    descriptors_b = np.array([[0, 0], [0, 255], [0, 240]], dtype=np.uint8)

    knn_queries, lowe_matches, mutual_matches = match_orb_descriptors(
        descriptors_a, descriptors_b, lowe_ratio=0.8
    )

    assert knn_queries == 3
    assert lowe_matches == 3
    assert [match.queryIdx for match in mutual_matches] == [0, 1]


def test_match_orb_descriptors_handles_missing_descriptors():
    assert match_orb_descriptors(None, np.array([[0]], dtype=np.uint8), 0.8) == (
        0,
        0,
        [],
    )
