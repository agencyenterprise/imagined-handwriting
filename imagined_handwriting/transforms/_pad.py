import numpy as np
from numpy import ndarray


class Pad:
    """Pads an array."""

    def __init__(self, pad_width: int, *, axis: int = 0):
        """Initialize a Pad transform.

        Args:
            pad_width: The number of timesteps to pad.
        """
        self.pad_width = pad_width
        self.axis = axis

    def __call__(self, a: ndarray) -> ndarray:
        """Pad the input array.

        Args:
            a: The input array.

        Returns:
            The padded array.
        """
        return pad(a, pad_width=self.pad_width, axis=self.axis)


def pad(a: ndarray, *, pad_width: int, axis: int = 0) -> ndarray:
    """Pads an array.

    Args:
        a: The array to pad.
        pad_width: The amount of padding to add to the front and
            back of the specified axis.

    Returns:
        The padded array.
    """
    padding = [(0, 0) for _ in range(np.ndim(a))]
    padding[axis] = (pad_width, pad_width)
    return np.pad(a, padding)
