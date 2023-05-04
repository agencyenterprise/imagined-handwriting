import pytest
import torch

from imagined_handwriting.nn import PositionalEncoding


@pytest.mark.parametrize("learned", [True, False])
def test_positional_encoding_returns_correct_shape(learned):
    """Test that the positional encoding returns the correct shape."""
    pe = PositionalEncoding(d_model=5, learned=learned)
    x = torch.randn(3, 10, 5)
    y = pe(x)
    assert y.shape == (3, 10, 5)


@pytest.mark.parametrize("learned", [True, False])
def test_encodings_are_added_to_input(learned):
    """Test that the positional encoding is added to the input."""
    pe = PositionalEncoding(d_model=5, learned=learned)
    x = torch.randn(3, 10, 5)
    y = pe(x)
    assert not torch.allclose(x, y)
