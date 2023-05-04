import pytest
import torch
from torch import nn

from imagined_handwriting.models import HandwritingTransformer
from imagined_handwriting.nn import FNetEncoderLayer


@pytest.mark.parametrize("attention", ["self", "fft"])
def test_transformer_returns_correct_shape(attention):
    """Test that the transformer returns the correct shape."""
    model = HandwritingTransformer(
        input_channels=5, num_classes=20, attention=attention
    )
    x = torch.randn(3, 10, 5)
    y_start, y_char = model(x)
    assert y_start.shape == (3, 10)
    assert y_char.shape == (3, 10, 19)


@pytest.mark.parametrize("attention", ["self", "fft"])
def test_transformer_uses_multiple_sessions(attention):
    """Test that the transformer uses multiple sessions."""
    model = HandwritingTransformer(
        input_channels=5,
        num_classes=20,
        num_layers=4,
        session_ids=["a", "b"],
        attention=attention,
    )
    model.eval()  # make forward pass deterministic
    x = torch.randn(3, 10, 5)
    a = [model(x, session_id="a") for _ in range(3)]
    b = [model(x, session_id="b") for _ in range(3)]
    for i in range(1, 3):
        torch.testing.assert_close(a[0][0], a[i][0])
        torch.testing.assert_close(a[0][1], a[i][1])
        torch.testing.assert_close(b[0][0], b[i][0])
        torch.testing.assert_close(b[0][1], b[i][1])
    assert not torch.allclose(a[0][0], b[0][0])
    assert not torch.allclose(a[0][1], b[0][1])


@pytest.mark.parametrize("attention", ["self", "fft"])
@pytest.mark.parametrize("session_id", ["c", None])
def test_transformer_raises_when_session_id_is_missing(attention, session_id):
    """Test that the transformer which was configured with session ids
    raises when a session id is missing."""
    model = HandwritingTransformer(
        input_channels=5,
        num_classes=20,
        num_layers=4,
        session_ids=["a", "b"],
        attention=attention,
    )
    x = torch.randn(3, 10, 5)
    with pytest.raises((ValueError, KeyError)):
        _ = model(x, session_id=session_id)


@pytest.mark.parametrize("attention", ["self", "fft"])
def test_transformer_has_correct_layer_type(attention):
    """Test that the transformer has the correct layer type."""
    model = HandwritingTransformer(
        input_channels=5,
        num_classes=20,
        num_layers=4,
        session_ids=["a", "b"],
        attention=attention,
    )
    for layer in model.encoder.layers:
        if attention == "self":
            assert isinstance(layer, nn.TransformerEncoderLayer)
        elif attention == "fft":
            assert isinstance(layer, FNetEncoderLayer)
