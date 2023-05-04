import numpy as np
import pytest

from imagined_handwriting.transforms._timewarp import random_timewarp, timewarp


@pytest.mark.parametrize(
    "new_timesteps,expected",
    [
        (1, np.array([1])),
        (2, np.array([1, 5])),
        (3, np.array([1, 3, 5])),
        (4, np.array([1, 2, 4, 5])),
        (5, np.array([1, 2, 3, 4, 5])),
        (10, np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])),
        (11, np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5])),
    ],
)
def test_timewarp_warps_correctly(new_timesteps, expected):
    """Verifies the warping is correct when the new timesteps."""
    a = np.array([1, 2, 3, 4, 5])
    actual = timewarp(a, new_timesteps)
    np.testing.assert_array_equal(actual, expected)


def test_random_timewarp_warps_correctly():
    """Verifies that a random timewarp warps correctly.

    We discover the correct number of timesteps by calling the
    rng function with a known seed.
    """
    SEED = 0
    rng = np.random.default_rng(SEED)

    a = np.array([1, 2, 3, 4, 5])
    actual = random_timewarp(a, low=1, high=2, rng=rng)
    expected = np.array([1, 2, 2, 3, 3, 4, 4, 5])
    np.testing.assert_array_equal(actual, expected)


def test_timewarp_handles_zero_newtimesteps():
    """Verifies that when new timesteps is zero, the tensor is empty"""
    t = np.array([1, 2, 3, 4, 5])
    new_timesteps = 0

    actual = timewarp(t, new_timesteps)
    expected = np.array([])
    np.testing.assert_array_equal(actual, expected)
