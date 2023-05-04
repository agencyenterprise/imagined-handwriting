import numpy as np

from imagined_handwriting.transforms import _mask


def test_mask_before_first_character_masks():
    """Verify mask_before_first_character masks the correct indices"""
    loss_mask = np.ones(10)
    y_start = np.ones(10)
    y_start[:3] = 0
    expected = np.ones_like(loss_mask)
    expected[:3] = 0

    actual = _mask.mask_loss_before_first_character(loss_mask, y_start)
    np.testing.assert_array_equal(actual, expected)


def test_mask_before_first_character_when_start_is_first_index():
    """Verify mask_before_first_character masks until first complete character"""
    loss_mask = np.ones(10)
    y_start = np.ones(10)
    y_start[3:5] = 0  # character start from 0-2, 5-9
    expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    actual = _mask.mask_loss_before_first_character(loss_mask, y_start)
    np.testing.assert_array_equal(actual, expected)


def test_mask_blank_window_masks():
    """Verify mask_blank_window masks the correct indices"""
    loss_mask = np.ones(10).reshape(1, 10)
    blank_windows = [[np.array([0, 1, 2]), np.array([7, 8])]]
    expected = np.array([[0, 0, 0, 1, 1, 1, 1, 0, 0, 1]])
    actual = _mask.mask_loss_on_blank_windows(loss_mask, blank_windows)
    np.testing.assert_array_equal(actual, expected)


def test_mask_blank_window_handles_empty_windows():
    """Verify mask_blank_window handles empty windows"""
    loss_mask = np.ones(10).reshape(1, 10)
    blank_windows = [[np.array([]), np.array([])]]
    expected = np.ones_like(loss_mask)
    actual = _mask.mask_loss_on_blank_windows(loss_mask, blank_windows)
    np.testing.assert_array_equal(actual, expected)


def test_mask_after_last_character():
    """Verify mask_after_last_character masks the correct indices"""
    loss_mask = np.ones(10).reshape(1, -1)
    expected = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).reshape(1, -1)

    actual = _mask.mask_loss_after_last_character(loss_mask, np.array([7]))
    np.testing.assert_array_equal(actual, expected)
