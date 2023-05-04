from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator


class RandomTimeWarp:
    """Randomly time warps a tensor."""

    def __init__(
        self,
        *,
        low: float,
        high: float,
        rng: Optional[Generator] = None,
    ):
        """Initialize a RandomTimewarp transform.

        Args:
            low: The minimum timewarp factor.
            high: The maximum timewarp factor
            rng: An optional numpy random number generator.
        """
        self.low = low
        self.high = high
        self.rng = rng or np.random.default_rng()

    def __call__(self, a: ndarray) -> ndarray:
        """Time warp the input array.

        Args:
            a: The input array.

        Returns:
            The time warped array.
        """
        return random_timewarp(a, low=self.low, high=self.high, rng=self.rng)


class TimeWarp:
    """Time warps a tensor."""

    def __init__(self, *, new_timesteps: int):
        """Initialize a Timewarp transform.

        Args:
            new_timesteps: The number of new timesteps in the warped tensor.
        """
        self.new_timesteps = new_timesteps

    def __call__(self, a: ndarray) -> ndarray:
        """Time warp the input array.

        Args:
            a: The input array.

        Returns:
            The time warped array.
        """
        return timewarp(a, new_timesteps=self.new_timesteps)


def random_timewarp(
    a: ndarray, *, low: float, high: float, rng: Optional[Generator] = None
) -> ndarray:
    """Randomly time warps a tensor.

    Args:
        a: A tensor of shape (time, ...).
        low: The minimum timewarp factor.
        high: The maximum timewarp factor
        rng: An optional numpy random number generator.

    Returns:
        The tensor that has been time warped via nearest neighbors interpolation.
    """
    if rng is None:
        rng = np.random.default_rng()

    original_timesteps = a.shape[0]
    timestep_multiplier = rng.uniform(low=low, high=high)
    new_timesteps = int(np.floor(original_timesteps * timestep_multiplier))
    return timewarp(a, new_timesteps)


def timewarp(a: ndarray, new_timesteps: int) -> ndarray:
    """Timewarps an array by stretching or shrinking via nearest-neighbor interpolation.

    Args:
        a: A numpy array with time along axis 0.
        new_timesteps: The number of new timesteps in the warped tensor.

    Return:
        A numpy array which is interpolated to the size of new timesteps.
    """
    input_timesteps = a.shape[0]
    index = np.linspace(0, input_timesteps - 1, new_timesteps)
    index = np.round(index).astype(int)
    return a[index]
