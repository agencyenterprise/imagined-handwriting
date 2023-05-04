import numpy as np
from numpy import ndarray


class Bin:
    """Bins an array."""

    def __init__(
        self,
        width: int,
        *,
        method: str,
        axis: int = 0,
    ):
        """Initialize a Bin transform.

        Args:
            width (int): The width of the bin.
            method (str): The method to use for binning. Must be a
                valid numpy function with an axis parameters.  E.g.
                "mean", "std", "sum", "max", "min", etc.
            axis (int, optional): The axis to bin along. Defaults to 0.
        """
        self.width = width
        self.method = method
        self.axis = axis

    def __call__(self, a: ndarray) -> ndarray:
        """Bin the input array.

        Args:
            a (np.ndarray): The input array.

        Returns:
            np.ndarray: The binned array.
        """
        if self.width < 2:
            return a
        return bin(a, width=self.width, method=self.method, axis=self.axis)


def bin(data: ndarray, width: int, method: str, axis: int = 0) -> ndarray:
    """Bins the data along the specified axis with the specified method.

    If the data is not divisible by `width` along the specified axis
    then the data will be truncated.

    Args:
        data (np.ndarray): The data to bin.
        width (int): The width of the bin.
        method (str): The method to use for binning. Must be a
            valid numpy function with an axis parameters.  E.g.
            "mean", "std", "sum", "max", "min", etc.
        axis (int, optional): The axis to bin along. Defaults to 0.


    Returns:
        np.ndarray: The binned data.
    """
    if not hasattr(np, method):
        raise ValueError(f"Invalid method: {method}")

    shape_tuple = [slice(None)] * data.ndim
    shape_tuple[axis] = slice(0, data.shape[axis] - data.shape[axis] % width)
    data = data[tuple(shape_tuple)]

    reshape_tuple = []
    for i in range(data.ndim):
        if i == axis:
            reshape_tuple.append(data.shape[i] // width)
            reshape_tuple.append(width)
        else:
            reshape_tuple.append(data.shape[i])

    reduce = getattr(np, method)
    return reduce(data.reshape(reshape_tuple), axis=axis + 1)
