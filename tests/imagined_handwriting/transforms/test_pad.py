import numpy as np
import pytest

from imagined_handwriting.transforms._pad import pad


@pytest.mark.parametrize("axis", [0, 1])
def test_pad_pads_correct_axis(axis):
    a = np.zeros((10, 10))
    a = pad(a, pad_width=2, axis=axis)
    for i in range(np.ndim(a)):
        if i == axis:
            assert a.shape[i] == 14
        else:
            assert a.shape[i] == 10
