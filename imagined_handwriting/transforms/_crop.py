from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray
from numpy.random import Generator


class RandomCrop:
    """Randomly crop an array."""

    def __init__(
        self,
        *,
        size: int,
        low=None,
        high=None,
        axis: int = 0,
        rng: Optional[Generator] = None,
        keys: Optional[List[Any]] = None,
    ):
        """Initialize a RandomCrop transform.

        Args:
            size (int): The size of the crop.
            low (int | None): The minimum value for the crop. If None, then
                the crop will start at the beginning of the array.
            high (int | None): The maximum value for the crop. If None, then
                the crop will end at the end of the array.
            axis (int): The axis to crop along.
            rng (Generator | None): An optional random number generator.
            keys (List[Any] | None): An optional list of keys to crop. Only
                used if the input is a dictionary.
        """
        self.size = size
        self.low = low
        self.high = high
        self.axis = axis
        self.rng = rng or np.random.default_rng()
        self.keys = keys

    def __call__(
        self, a: Union[ndarray, Dict[Any, ndarray], Sequence[ndarray]]
    ) -> Union[ndarray, Dict[Any, ndarray], Sequence[ndarray]]:
        """Crop the input array.

        Args:
            a (ndarray | Dict[Any, ndarray] | Sequence[ndarray]): The input array.

        Returns:
            (ndarray | Dict[Any, ndarray] | Sequence[ndarray]): The cropped array.
        """
        bounds = self._get_bounds(a)
        return crop(a, bounds, axis=self.axis)

    def _get_bounds(
        self, a: Union[ndarray, dict, Sequence[ndarray]]
    ) -> Tuple[int, int]:
        """Get the bounds for the crop.

        Args:
            a (ndarray | Dict[Any, ndarray] | Sequence[ndarray]): The input array.

        Returns:
            (Tuple[int, int]): The bounds for the crop.
        """
        low = self.low or 0
        high = self.high or self.axis_len(a)

        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")

        if high - low < self.size:
            raise ValueError(
                f"size ({self.size}) must be less than the length bounds interval "
                f"along axis {self.axis} ({high - low})"
            )

        start = self.rng.integers(low=low, high=high - self.size)
        stop = start + self.size
        return start, stop

    def axis_len(self, a: Union[ndarray, dict, Sequence[ndarray]]) -> int:
        """Get the length of the axis.

        Args:
            a (ndarray | Dict[Any, ndarray] | Sequence[ndarray]): The input array.

        Returns:
            (int): The length of the axis.
        """
        return _axis_len(a, self.axis)


def random_crop(
    x: Union[ndarray, dict, Sequence[ndarray]],
    crop_size: int,
    *,
    axis: int = 0,
    keys=Optional[List[Any]],
    rng: Optional[Generator] = None,
) -> Union[ndarray, dict, Sequence[ndarray]]:
    """Randomly crop an array or dictionary of arrays along the specified axis

    Args:
        x (ndarray or dict): The array or dictionary of arrays that will be cropped.
        axis (int): The axis to perform the crop.
        keys (List[Any] | None): An optional list of keys to crop. Only
            used if the input is a dictionary.

    Returns:
        (ndarray or dict): The cropped array or dictionary of arrays.
            The size of the provided axis will be bounds[1] - bounds[0].

    """
    if rng is None:
        rng = np.random.default_rng()
    low = 0
    high = _axis_len(x, axis, keys)

    start = rng.integers(low=low, high=high - crop_size)
    end = start + crop_size
    bounds = (start, end)
    return crop(x=x, bounds=bounds, axis=axis, keys=keys)


def crop(
    x: Union[ndarray, dict, Sequence[ndarray]],
    bounds: Tuple[int, int],
    axis: int = 0,
    keys: Optional[List[Any]] = None,
) -> Union[ndarray, dict, Sequence[ndarray]]:
    """Crop an array or dictionary of arrays along the specified axis

    Args:
        x (ndarray or dict): The array or dictionary of arrays that will be cropped.
        bounds (int,int): The lower and upper indices for the crop.
            Everything between these indices will be kept and
            everything outside of them will be cropped.  Note that
            the lower bound is inclusive and the upper bound
            is exclusive.
        axis (int): The axis to perform the crop.

    Returns:
        (ndarray or dict): The cropped array or dictionary of arrays.
            The size of the provided axis will be bounds[1] - bounds[0].

    """
    if isinstance(x, dict):
        return crop_dict(x, bounds, axis, keys=keys)
    elif isinstance(x, ndarray):
        return crop_array(x, bounds, axis)
    elif isinstance(x, Sequence):
        seq_type = type(x)
        return seq_type([crop_array(a, bounds, axis) for a in x])  # type: ignore
    else:
        raise ValueError(f"Crop only supports ndarray and dict, not {type(x)}")


def crop_array(a: ndarray, bounds: Tuple[int, int], axis: int = 0) -> ndarray:
    """Crop an array along the specified axis

    Args:
        a (ndarray): The array that will be cropped.
        bounds (int,int): The lower and upper indices for the crop.
            Everything between these indices will be kept and
            everything outside of them will be cropped.  Note that
            the lower bound is inclusive and the upper bound
            is exclusive.
        axis (int): The axis to perform the crop.


    Returns:
        (ndarray): The cropped array.  The size of the provided axis will be
            bounds[1] - bounds[0].

    """
    slices = [slice(None)] * np.ndim(a)
    slices[axis] = slice(*bounds)
    return a[tuple(slices)]


def crop_dict(
    d: dict, bounds: Tuple[int, int], axis: int = 0, keys: Optional[List[Any]] = None
) -> dict:
    """Crop a dictionary of arrays along the specified axis

    Args:
        d (dict): The dictionary of arrays that will be cropped.
        bounds (int,int): The lower and upper indices for the crop.
            Everything between these indices will be kept and
            everything outside of them will be cropped.  Note that
            the lower bound is inclusive and the upper bound
            is exclusive.
        axis (int): The axis to perform the crop.
        keys(Optional[List[Any]]): The keys to crop.  If None then all keys
            will be cropped.

    Returns:
        (dict): The cropped dictionary.  The size of the provided axis will be
            bounds[1] - bounds[0].

    """
    if keys is None:
        keys = list(d.keys())
    for k in keys:
        if k not in d:
            raise KeyError(f"Key {k} not found in dictionary")
    cropped = {}
    for k in d.keys():
        if k in keys:
            cropped[k] = crop_array(d[k], bounds, axis)
        else:
            cropped[k] = d[k]
    return cropped


def _axis_len(
    a: Union[ndarray, dict, Sequence[ndarray]], axis: int, keys: Optional[List] = None
) -> int:
    """Get the high value for the crop.

    Args:
        a (ndarray | Dict[Any, ndarray] | Sequence[ndarray]): The input array.

    Returns:
        (int): The high value for the crop.
    """
    if isinstance(a, ndarray):
        return a.shape[axis]
    if isinstance(a, Sequence):
        return a[0].shape[axis]
    elif isinstance(a, dict):
        if keys is None:
            return next(iter(a.values())).shape[axis]
        else:
            return a[keys[0]].shape[axis]
