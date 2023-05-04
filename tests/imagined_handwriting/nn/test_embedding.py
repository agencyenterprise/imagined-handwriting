import pytest
import torch

from imagined_handwriting.nn import ChannelEmbedding


def test_channel_embedding_linear_returns_correct_shape():
    """Verifies that linear channel embedding returns the correct shape."""
    embedding = ChannelEmbedding(10, 20, linear=True)
    x = torch.randn(2, 3, 10)
    out = embedding(x)
    assert out.shape == (2, 3, 20)


def test_channel_embedding_tokens_returns_correct_shape():
    """Verifies that token channel embedding returns the correct shape."""
    embedding = ChannelEmbedding(10, 20, linear=False)
    x = torch.randint(10, (2, 3, 10))
    out = embedding(x)
    assert out.shape == (2, 3, 20)


def test_channel_embedding_tokens_errors_with_large_token():
    """Verifies that token channel embedding errors with large token."""
    embedding = ChannelEmbedding(10, 20, linear=False)
    x = torch.tensor([[[11]]])
    with pytest.raises(IndexError):
        _ = embedding(x)
