from __future__ import annotations

import numpy as np

from segmentation.utils import average_weights, weighted_average_weights


def test_average_weights_matches_manual_mean():
    weights = [
        [np.array([1.0, 3.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32)],
        [np.array([5.0, 7.0], dtype=np.float32), np.array([[3.0]], dtype=np.float32)],
        [np.array([9.0, 11.0], dtype=np.float32), np.array([[5.0]], dtype=np.float32)],
    ]

    averaged = average_weights(weights)

    np.testing.assert_allclose(averaged[0], np.array([5.0, 7.0], dtype=np.float32))
    np.testing.assert_allclose(averaged[1], np.array([[3.0]], dtype=np.float32))


def test_weighted_average_weights_matches_manual_weighting():
    weights = [
        [np.array([2.0, 4.0], dtype=np.float32), np.array([[1.0]], dtype=np.float32)],
        [np.array([8.0, 10.0], dtype=np.float32), np.array([[5.0]], dtype=np.float32)],
    ]
    counts = [1, 3]

    averaged = weighted_average_weights(weights, counts)

    np.testing.assert_allclose(averaged[0], np.array([6.5, 8.5], dtype=np.float32))
    np.testing.assert_allclose(averaged[1], np.array([[4.0]], dtype=np.float32))
