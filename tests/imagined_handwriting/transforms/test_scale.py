import numpy as np

from imagined_handwriting.transforms._scale import random_scale


def test_random_scale_is_deterministic_with_generator():
    """Verify that random_scale is deterministic with a generator"""
    a = np.ones((10, 9, 8))
    low = 0.5
    high = 1.5
    rng = np.random.default_rng(0)
    first = random_scale(a, low=low, high=high, rng=rng)
    rng = np.random.default_rng(0)
    second = random_scale(a, low=low, high=high, rng=rng)
    np.testing.assert_array_equal(first, second)
