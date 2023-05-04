import torch

from imagined_handwriting.nn import DimensionDropout


def test_dimension_dropout_returns_correct_shape():
    """Test that the dimension dropout returns the correct shape."""
    dropout = DimensionDropout(0.5, dim=1)
    x = torch.randn(3, 10, 5)
    y = dropout(x)
    assert y.shape == (3, 10, 5)


def test_dimension_dropout_does_not_error_with_degenerate_dimension():
    """Test that the dimension dropout handles a degenerate dimension."""
    dropout = DimensionDropout(0.5, dim=1)
    x = torch.randn(3, 1)
    _ = dropout(x)


def test_dimension_dropout_is_off_for_eval():
    """Test that the dimension dropout is off during evaluation."""
    dropout = DimensionDropout(0.5, dim=1)
    dropout.eval()
    x = torch.randn(3, 10, 5)
    y = dropout(x)
    torch.testing.assert_close(x, y)
