import numpy as np
import pytest

from imagined_handwriting.transforms import _noise


@pytest.mark.parametrize(
    "axis,index", [(0, (0, slice(None), slice(None))), ((0, 2), (0, slice(None), 0))]
)
def test_add_offset_noise_is_constant_along_dim(axis, index):
    """Verify that the offset noise is constant along non-noise axes"""
    data = np.ones((10, 9, 8))
    std = 1.0
    actual = _noise.offset_noise(data, std, axis)[index]
    # slice into the axes that were not specified for noise
    # the value should be constant there
    assert np.all(actual[0] == actual)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_random_walk_noise_completes_along_each_dim(axis):
    """Verify that add_random_walk_noise does not error

    The purpose of this is test is to verify that all shapes are correct
    and the noise is broadcastable so it actually completes.
    """
    data = np.ones((10, 9, 8))
    std = 1
    _ = _noise.random_walk_noise(data, std, axis)
