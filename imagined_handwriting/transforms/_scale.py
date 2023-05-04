from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator as Generator


class RandomScale:
    """Randomly scales an array."""

    def __init__(
        self,
        *,
        low: float,
        high: float,
        rng: Optional[Generator] = None,
    ):
        """Initialize a RandomScale transform.

        Args:
            low: The lower bound of the scaling multiplier.
            high: The upper bound of the scaling multiplier.
            rng: An optional numpy random number generator.
        """
        self.low = low
        self.high = high
        self.rng = rng or np.random.default_rng()

    def __call__(self, a: ndarray) -> ndarray:
        """Scale the input array.

        Args:
            a: The input array.

        Returns:
            The scaled array.
        """
        return random_scale(a, low=self.low, high=self.high, rng=self.rng)


class Scale:
    """Scales an array."""

    def __init__(self, *, multiplier: float):
        """Initialize a Scale transform.

        Args:
            multiplier: The scaling multiplier.
        """
        self.multiplier = multiplier

    def __call__(self, a: ndarray) -> ndarray:
        """Scale the input array.

        Args:
            a: The input array.

        Returns:
            The scaled array.
        """
        return scale(a, multiplier=self.multiplier)


def random_scale(
    a: ndarray, *, low: float, high: float, rng: Optional[Generator] = None
) -> ndarray:
    """Scales an array by a random multiplier.


    Args:
        a: The array to scale.
        low: The lower bound of the scaling multiplier.
        high: The upper bound of the scaling multiplier.
        rng: An optional numpy random number generator.

    Returns:
        The scaled array.

    """
    if rng is None:
        rng = np.random.default_rng()
    return a * rng.uniform(low=low, high=high)


def scale(a: ndarray, *, multiplier: float) -> ndarray:
    """Scales an array by a multiplier.

    Args:
        a: The array to scale.
        multiplier: The scaling multiplier.

    Returns:
        The scaled array.

    """
    return a * multiplier
