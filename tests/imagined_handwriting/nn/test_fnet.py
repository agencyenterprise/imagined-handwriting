import pytest
import torch

from imagined_handwriting.nn import FNetEncoderLayer


def test_fnet_encoder_layer_returns_correct_shape():
    """Test that the fnet encoder layer returns the correct shape."""
    layer = FNetEncoderLayer(d_model=5, dim_feedforward=5)
    x = torch.randn(3, 10, 5)
    y = layer(x)
    assert y.shape == (3, 10, 5)


def test_fnet_encoder_layer_raises_with_src_mask():
    """Test that the fnet encoder layer raises with a src_mask."""
    layer = FNetEncoderLayer(d_model=5, dim_feedforward=5)
    x = torch.randn(3, 10, 5)
    with pytest.raises(ValueError):
        _ = layer(x, src_mask=torch.ones(3, 10, 10))


def test_fnet_encoder_layer_raises_with_src_key_padding_mask():
    """Test that the fnet encoder layer raises with a src_key_padding_mask."""
    layer = FNetEncoderLayer(d_model=5, dim_feedforward=5)
    x = torch.randn(3, 10, 5)
    with pytest.raises(ValueError):
        _ = layer(x, src_key_padding_mask=torch.ones(3, 10))
