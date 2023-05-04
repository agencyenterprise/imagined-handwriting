from typing import Optional, Sequence, Union

import numpy as np
from numpy import ndarray
from numpy.random import Generator


class WhiteNoise:
    """Add white noise to the input data."""

    def __init__(self, *, std: float, rng: Optional[Generator] = None):
        """Initialize a WhiteNoise transform.

        Args:
            std: The standard deviation of the normal distribution
                used to sample noise from (mean=0)
            rng: An optional numpy random number generator.
        """
        self.std = std
        self.rng = rng or np.random.default_rng()

    def __call__(self, a: ndarray) -> ndarray:
        """Add white noise to the input array.

        Args:
            a: The input array.

        Returns:
            The input array with added white noise.
        """
        return white_noise(a, std=self.std, rng=self.rng)


class OffsetNoise:
    """Add constant offset noise to the input data."""

    def __init__(
        self, *, std: float, axis: Union[int, Sequence], rng: Optional[Generator] = None
    ):
        """Initialize a OffsetNoise transform.

        Args:
            std: The standard deviation of the normal distribution
                used to sample noise from (mean=0)
            axis: The axis to apply the offset over.
            rng: An optional numpy random number generator.
        """
        self.std = std
        self.axis = axis
        self.rng = rng or np.random.default_rng()

    def __call__(self, a: ndarray) -> ndarray:
        """Add constant offset noise to the input array.

        Args:
            a: The input array.

        Returns:
            The input array with added constant offset noise.
        """
        return offset_noise(a, std=self.std, axis=self.axis, rng=self.rng)


class RandomWalkNoise:
    """Add random walk noise to the input data."""

    def __init__(self, *, std: float, axis: int, rng: Optional[Generator] = None):
        """Initialize a RandomWalkNoise transform.

        Args:
            std: The standard deviation of the normal distribution
                used to sample noise from (mean=0)
            axis: The axis to aggregate noise for the random walk
            rng: An optional numpy random number generator.
        """
        self.std = std
        self.axis = axis
        self.rng = rng or np.random.default_rng()

    def __call__(self, a: ndarray) -> ndarray:
        """Add random walk noise to the input array.

        Args:
            a: The input array.

        Returns:
            The input array with added random walk noise.
        """
        return random_walk_noise(a, std=self.std, axis=self.axis, rng=self.rng)


def white_noise(a: ndarray, std: float, rng: Optional[Generator] = None):
    """Add white noise to the input data.

    Args:
        x (ndarray): The input neural data of shape (timesteps, channels)
        std (float): The standard deviation of the normal distribution
            used to sample noise from (mean=0)
        rng (Generator | None): An optional random number generator.

    Returns:
        The input data with added white noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=std, size=a.shape)
    return a + noise


def offset_noise(
    a: ndarray,
    std: float,
    axis: Union[int, Sequence] = 0,
    rng: Optional[Generator] = None,
) -> ndarray:
    """Add constant offset over the specified axis

    This function will add random constant offset over the specified axes.
    The use case we have in mind is a batch of (trial, time, electrode) where
    we will add the offset over axis=(0,2) so that the offset for a fixed
    electrode and fixed sentence is constant over time.

    Args:
        a (ndarray): The array to add a constant offset to
        std (flot): The standard deviation for sampling the offset noise
        axis (int or Sequence): The axis to apply the offset over.
        rng (Generator | None): An optional random number generator.

    Returns
        (ndarray): The input array with added offset noise
    """
    if rng is None:
        rng = np.random.default_rng()

    n_dim = np.ndim(a)
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, Sequence):
        axis = tuple(axis)

    size = [1 for _ in range(n_dim)]
    for ax in axis:
        size[ax] = a.shape[ax]

    offset_noise = rng.normal(loc=0, scale=std, size=size)

    return a + offset_noise


def random_walk_noise(
    a: ndarray, std: float, axis: int = 0, rng: Optional[Generator] = None
) -> ndarray:
    """Add random walk noise to the input array

    Args:
        a (ndarray): The array to which noise is added.
        std (float): The standard deviation of the normal distribution
            used to sample noise from (mean=0)
        axis (int): The axis to aggregate noise for the random walk
        rng (Generator | None): An optional random number generator.

    Returns:
        (ndarray) The array with added random walk noise along the specified axis.

    """
    if rng is None:
        rng = np.random.default_rng()
    walk_noise = np.cumsum(rng.normal(loc=0, scale=std, size=a.shape), axis=axis)
    return a + walk_noise
