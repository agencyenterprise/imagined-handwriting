import numpy as np

from imagined_handwriting.transforms._smooth import smooth


def test_smooth_applies_to_correct_axis():
    """Verifies that smoothing only applies to a single channel at a time."""
    x = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
    y = smooth(x, kernel_width=3)
    assert all(y[0, 5:] == 0)
    assert all(y[1, :4] == 0)


def test_smooth_returns_original_datatype():
    """Verifies the smooth output datatype is the same as the input datatype."""
    x = np.random.randn(10, 10).astype(np.float32)
    y = smooth(x)
    assert y.dtype == x.dtype
