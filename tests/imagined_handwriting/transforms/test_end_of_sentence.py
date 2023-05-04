import numpy as np

from imagined_handwriting.transforms._end_of_sentence import (
    get_last_character_start_index,
    remove_pause_from_end_of_sentence,
)


def test_remove_pause_from_end_of_sentence_when_pause():
    """Verify adjust_end_of_sentence_for_max_pause is correct when there is a pause."""
    end_of_sentence = np.array([10])
    y_start = np.array([1, 0, 0, 0, 0]).reshape(1, -1)
    max_pause = 2
    expected = np.array([2])
    actual = remove_pause_from_end_of_sentence(end_of_sentence, y_start, max_pause)
    np.testing.assert_array_equal(actual, expected)


def test_remove_pause_from_end_of_sentence_returns_eos_when_no_pause():
    """Verify adjust_end_of_sentence_for_max_pause returns eos when no pause"""
    end_of_sentence = np.array([3])
    y_start = np.array([1, 0, 0, 0, 0]).reshape(1, -1)
    max_pause = 10
    expected = np.array([3])
    actual = remove_pause_from_end_of_sentence(end_of_sentence, y_start, max_pause)
    np.testing.assert_array_equal(actual, expected)


def test_get_last_character_start_index():
    """Verify get_last_character_start_index is correct on 1d array of single trial"""
    data = np.array([0, 0, 1, 0, 1, 0]).reshape(1, -1)
    expected = np.array([4])
    actual = get_last_character_start_index(data)
    np.testing.assert_array_equal(actual, expected)


def test_get_last_character_start_index_2d():
    """Verify get_last_character_start_index is correct on multiple trials"""
    data = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0]])
    expected = np.array([5, 4])
    actual = get_last_character_start_index(data)
    np.testing.assert_array_equal(actual, expected)


def test_get_last_character_index_handles_no_start_times():
    """Verify get_last_character_index handles no start times"""
    data = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    expected = np.array([0, 0])
    actual = get_last_character_start_index(data)
    np.testing.assert_array_equal(actual, expected)
