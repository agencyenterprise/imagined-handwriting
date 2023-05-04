import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imagined_handwriting.transforms._bin import bin


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_bin_returns_correct_shape(axis):
    """Tests that bin returns the correct shape"""
    x = np.random.rand(100, 100, 100)
    actual = bin(x, 5, "mean", axis=axis).shape
    expected = tuple(100 if i != axis else 20 for i in range(3))
    assert actual == expected


@pytest.mark.parametrize("method", ["mean", "sum", "max", "min"])
def test_bin_returns_correct_values(method):
    """Tests that bin returns the correct values"""
    x = np.random.rand(100, 100, 100)
    actual = bin(x, 5, method, axis=0)
    expected = getattr(np, method)(x.reshape(20, 5, 100, 100), axis=1)
    assert_array_equal(actual, expected)


def test_bin_is_noop_for_width_less_than_2():
    """Tests that bin is a noop for width less than 2"""
    x = np.random.rand(100, 100, 100)
    actual = bin(x, 1, "mean", axis=0)
    assert_array_equal(actual, x)
