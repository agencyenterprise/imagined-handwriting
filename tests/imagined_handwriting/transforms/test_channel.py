import numpy as np

from imagined_handwriting.transforms._channel import channel_to_embedding_index


def test_channel_to_embedding_index_returns_correct_indices():
    """Verifies that the channel to embedding index function
    returns the correct indices."""
    x = np.array([[[0, 0, 0], [1, 1, 1]]])
    actual = channel_to_embedding_index(x, 1)
    expected = np.array([[[0, 2, 4], [1, 3, 5]]])
    assert np.array_equal(actual, expected)
