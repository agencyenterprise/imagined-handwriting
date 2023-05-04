import numpy as np
import pytest

from imagined_handwriting.transforms._crop import (
    RandomCrop,
    crop,
    crop_array,
    crop_dict,
)


def test_random_crop_returns_correct_shape():
    """Verify RandomCrop returns the correct shape"""
    data = np.arange(10)
    actual = RandomCrop(size=2)(data).shape
    expected = (2,)
    assert actual == expected


def test_random_crop_raises_when_size_is_larger_than_bounds():
    """Verify RandomCrop raises an error when size is larger than bounds"""
    data = np.arange(10)
    with pytest.raises(ValueError):
        RandomCrop(size=20)(data)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("input", ["array", "dict", "sequence"])
def test_random_crop_infers_correct_axis_size(axis, input):
    """Verify RandomCrop infers the correct bounds"""
    data = {
        "array": np.random.rand(10, 20, 30),
        "dict": {"a": np.random.rand(10, 20, 30), "b": np.random.rand(10, 20, 30)},
        "sequence": [np.random.rand(10, 20, 30), np.random.rand(10, 20, 30)],
    }

    data = data[input]
    expected = [10, 20, 30][axis]
    actual = RandomCrop(size=5, axis=axis).axis_len(data)
    assert actual == expected


def test_crop_array_crops_to_correct_size():
    """Verify the cropped array is the correct size"""
    data = np.arange(10)
    bounds = (3, 5)
    dim = 0
    expected = np.array([3, 4])

    actual = crop_array(data, bounds, dim)
    np.testing.assert_array_equal(actual, expected)


def test_crop_array_crops_along_dim():
    """Verify crop_array works on 2D array along the second dim"""
    data = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    bounds = (1, 3)
    dim = 1
    expected = np.array([[1, 2], [5, 6]])

    actual = crop_array(data, bounds, dim)
    np.testing.assert_array_equal(actual, expected)


def test_crop_dict_crops_all_arrays():
    """Verify crop_dict works on a dictionary of arrays"""
    data = {"a": np.arange(10), "b": np.arange(10, 20)}
    bounds = (3, 5)
    dim = 0
    expected = {"a": np.array([3, 4]), "b": np.array([13, 14])}

    actual = crop_dict(data, bounds, dim)
    for k in actual:
        np.testing.assert_array_equal(actual[k], expected[k])
    for k in expected:
        np.testing.assert_array_equal(actual[k], expected[k])


def test_crop_dict_respects_keys():
    """Verify crop_dict only crops arrays in the keys"""
    data = {"a": np.arange(10), "b": np.arange(10, 20)}
    bounds = (3, 5)
    dim = 0
    keys = ["a"]
    expected = {"a": np.array([3, 4]), "b": np.arange(10, 20)}

    actual = crop_dict(data, bounds, dim, keys)
    for k in actual:
        np.testing.assert_array_equal(actual[k], expected[k])
    for k in expected:
        np.testing.assert_array_equal(actual[k], expected[k])


def test_crop_dispatches_to_array():
    """Verify crop dispatches to crop_array when given an array"""
    data = np.arange(10)
    bounds = (3, 5)
    dim = 0
    expected = np.array([3, 4])

    actual = crop(data, bounds, dim)
    np.testing.assert_array_equal(actual, expected)


def test_crop_dispatches_to_dict():
    """Verify crop dispatches to crop_dict when given a dictionary"""
    data = {"a": np.arange(10), "b": np.arange(10, 20)}
    bounds = (3, 5)
    dim = 0
    expected = {"a": np.array([3, 4]), "b": np.array([13, 14])}

    actual = crop(data, bounds, dim)
    for k in actual:
        np.testing.assert_array_equal(actual[k], expected[k])
    for k in expected:
        np.testing.assert_array_equal(actual[k], expected[k])


@pytest.mark.parametrize("seq", [list, tuple])
def test_crop_returns_correct_sequence_type(seq):
    """Verify crop dispatches to crop_sequence when given a sequence"""
    data = seq([np.arange(10), np.arange(10, 20)])
    bounds = (3, 5)
    dim = 0

    actual = crop(data, bounds, dim)
    assert type(actual) == seq
