from typing import Optional

import numpy as np
from numpy import ndarray


def one_hot(a: ndarray, *, num_classes: Optional[int] = None):
    """Converts an array to a one-hot array.

    Args:
        a: The array to convert.
        num_classes: The number of classes in the one-hot array.
    Returns
        A one-hot array.
    """
    num_classes = a.max() + 1 if num_classes is None else num_classes
    return np.eye(num_classes)[a]


def sparse(a: ndarray, axis: int = -1):
    """Converts an array to a sparse array.

    Args:
        a: The array to convert.
    Returns
        A sparse array.
    """
    return a.argmax(axis=axis)
