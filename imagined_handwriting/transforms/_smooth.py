import numpy as np
from numpy import ndarray
from scipy import ndimage, signal


class Smooth:
    """Smooths data with a guassian filter.

    Args:
        kernel_width: The width of the gaussian kernel
        kernel_std: The standard deviation of the gaussian kernel
    """

    def __init__(self, kernel_width: int = 100, kernel_std: float = 2.0):
        self.kernel_width = kernel_width
        self.kernel_std = kernel_std

    def __call__(self, x: ndarray) -> ndarray:
        return smooth(x, self.kernel_width, self.kernel_std)


def smooth(x: ndarray, kernel_width: int = 100, kernel_std: float = 2.0) -> ndarray:
    """Smooths data with a guassian filter.

    Args:
        x: The input neural data of shape (timesteps, channels)

    Returns:
        the input data smoothed by a guassian filter
    """
    inpulse = np.zeros([kernel_width])
    inpulse[kernel_width // 2] = 1
    gauss_kernel = ndimage.gaussian_filter1d(inpulse, kernel_std)
    valid_idx = np.nonzero(gauss_kernel > 0.01)[0]
    gauss_kernel = gauss_kernel[valid_idx]
    filter = gauss_kernel / sum(gauss_kernel)
    return np.array(
        [signal.convolve(channel, filter, mode="same") for channel in x.T],
        dtype=x.dtype,
    ).T
