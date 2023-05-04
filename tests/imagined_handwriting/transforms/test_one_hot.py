import numpy as np
import pytest

from imagined_handwriting.transforms._one_hot import one_hot, sparse


@pytest.fixture
def one_hot_2d():
    one_hot = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    sparse = np.array([1, 0])
    return one_hot, sparse


@pytest.fixture
def one_hot_3d():
    one_hot = np.array([[[0, 1, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 0]]])
    sparse = np.array([[1, 0], [3, 2]])
    return one_hot, sparse


def test_one_hot_to_sparse_on_tensor_2d(one_hot_2d):
    """Verifies that converting to sparse labels works on 2d tensor."""
    data, expected = one_hot_2d
    actual = sparse(data)

    np.testing.assert_array_equal(actual, expected)


def test_one_hot_to_sparse_on_tensor_3d(one_hot_3d):
    """Verifies that converting to sparse labels works on 3d tensor."""
    data, expected = one_hot_3d
    actual = sparse(data)

    np.testing.assert_array_equal(actual, expected)


def test_sparse_to_one_hot_on_tensor_2d(one_hot_2d):
    """Verifies that converting to one hot labels works on 2d tensor."""
    expected, data = one_hot_2d
    actual = one_hot(data, num_classes=4)

    np.testing.assert_array_equal(actual, expected)


def test_sparse_to_one_hot_on_tensor_3d(one_hot_3d):
    """Verifies that converting to one hot labels works on 3d tensor."""
    expected, data = one_hot_3d
    actual = one_hot(data, num_classes=4)

    np.testing.assert_array_equal(actual, expected)
