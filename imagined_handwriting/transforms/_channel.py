from typing import Optional

import numpy as np
from numpy import ndarray


def clip_to_max_firing_rate(x: ndarray, max_firing_rate: int) -> ndarray:
    """Clip the firing rate of each channel to the maximum firing rate.

    Args:
        x: The input neural data.
        max_firing_rate: The maximum firing rate of each channel

    Returns:
        the input data with each channel clipped to the maximum firing rate
    """
    return np.clip(x, a_min=None, a_max=max_firing_rate)


def channel_to_embedding_index(
    x: ndarray, max_firing_rate: Optional[int] = None
) -> ndarray:
    """Maps channel values to embedding indices.

    The input data is expected to be integer firing rate counts
    for each channel.  This function then maps the firing rate
    of each channel to channel_index * max_firing_rate + firing_rate.
    which gives a unique index for each channel and firing rate.

    This allows us to embed the neural data into a continuous space
    in a way that preserves the channel information.

    Note: there is probably a more efficient implementation but it
    only needs to be done once during preprocessing.

    Args:
        x: The input neural data of shape (trials, timesteps, channels)
            of integer spike counts.
        max_firing_rate: The maximum firing rate allowed across all data.
            The firing rates will be clipped to this max value before
            creating the embedding indices.

    Returns:
        An array of shape (trials, timesteps, channels) of embedding indices.
    """
    if max_firing_rate is None:
        max_firing_rate = x.max()
    else:
        x = clip_to_max_firing_rate(x, max_firing_rate)

    map_array = np.arange(x.shape[2] * (max_firing_rate + 1)).reshape(
        -1, max_firing_rate + 1
    )
    new_x = np.zeros_like(x)
    for i in range(x.shape[0]):
        for t in range(x.shape[1]):
            for c in range(x.shape[2]):
                new_x[i, t, c] = map_array[c, x[i, t, c]]

    return new_x
