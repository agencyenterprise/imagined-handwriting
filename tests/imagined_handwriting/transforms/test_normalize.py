import numpy as np

from imagined_handwriting.transforms import _normalize


def test_normalize_by_block_normalizes():
    """Verifies that normalize by block normalizes each trial."""
    neural_activity = np.zeros((5, 4, 3))
    block_means = -np.ones((2, 3))
    sent_blocks = np.array([1, 1, 2, 2, 3])
    char_blocks = np.array([0, 4])
    std = 2 * np.ones(3)

    actual = _normalize.normalize(
        neural_activity,
        sentence_blocks=sent_blocks,
        character_blocks=char_blocks,
        block_means=block_means,
        block_std=std,
    )
    expected = 0.5 * np.ones((5, 4, 3))

    np.testing.assert_array_equal(actual, expected)


def test_closest_block_return_int():
    """Verifies that closest_block returns an integer type."""
    actual = _normalize.closest_block(10, np.array([1, 2, 3]))
    assert isinstance(actual, int)


def test_closest_block_returns_exact_match():
    """Verifies that the closest block returns the correct index for an exact match."""
    trial_block = 10
    char_blocks = np.array([9, 10, 11])

    actual = _normalize.closest_block(trial_block, char_blocks)
    expected = 1

    assert actual == expected


def test_closest_block_when_trial_is_large():
    """Verifies the closest block is the largest possible block when the
    trial block is larger than the largest character block."""
    trial_block = 100
    char_blocks = np.array([5, 6, 7])

    actual = _normalize.closest_block(trial_block, char_blocks)
    expected = 2

    assert actual == expected
