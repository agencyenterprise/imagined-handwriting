import pytest
import torch

from imagined_handwriting.nn import Downsample, Upsample


@pytest.mark.parametrize("factor", [1, 2, 3, 4, 24])
@pytest.mark.parametrize("method", ["mean", "drop"])
def test_downsample_returns_correct_shape(factor, method):
    """Test that the downsample returns the correct shape."""
    downsample = Downsample(factor, method)
    x = torch.randn(3, 24, 5)
    y = downsample(x)
    assert y.shape == (3, 24 // factor, 5)


@pytest.mark.parametrize("method", ["mean", "drop"])
def test_downsample_raises_error_with_non_positive_factor(method):
    """Test that the downsample raises an error with a non-positive factor."""
    with pytest.raises(ValueError):
        _ = Downsample(0, method)


@pytest.mark.parametrize("method", ["mean", "drop"])
def test_downsample_raises_when_factor_is_larger_than_sequence_length(method):
    """Test that the downsample raises an error when the factor is
    larger than the sequence length."""
    downsample = Downsample(100, method)
    x = torch.randn(3, 99, 5)
    with pytest.raises(ValueError):
        _ = downsample(x)


def test_downsample_raises_error_with_invalid_method():
    """Test that the downsample raises an error with an invalid method."""
    with pytest.raises(ValueError):
        _ = Downsample(2, "invalid")


def test_downsample_mean_is_computed_correctly():
    """Test that the downsample mean is computed correctly."""
    downsample = Downsample(2, "mean")
    x = torch.tensor([[[1, 2], [3, 4]]]).float()
    y = downsample(x)
    torch.testing.assert_close(y, torch.tensor([[[2, 3]]]).float())


def test_downsample_drop_is_computed_correctly():
    """Test that the downsample drop is computed correctly."""
    downsample = Downsample(2, "drop")
    x = torch.tensor([[[1, 2], [3, 4]]])
    y = downsample(x)
    torch.testing.assert_close(y, torch.tensor([[[1, 2]]]))


@pytest.mark.parametrize("factor", [1, 2, 3, 4])
def test_upsample_returns_correct_shape(factor):
    """Test that the upsample returns the correct shape."""
    upsample = Upsample(factor)
    x = torch.randn(3, 24, 5)
    y = upsample(x)
    assert y.shape == (3, 24 * factor, 5)
