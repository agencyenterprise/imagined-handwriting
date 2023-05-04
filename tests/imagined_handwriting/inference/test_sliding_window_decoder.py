import numpy as np
import pytest
import torch

from imagined_handwriting.inference.sliding_window_decoder import SlidingWindowDecoder


@pytest.fixture(autouse=True)
def n_chars():
    return 31


@pytest.fixture
def fake_decoder(n_chars):
    class FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_chars = n_chars

        def parameters(self):
            class Param:
                device = "cpu"

            yield Param()

        def forward(s, x):
            batch, time = x.shape[:2]
            return torch.zeros(batch, time), torch.zeros(batch, time, n_chars)

    return FakeDecoder()


@pytest.mark.parametrize("prediction_start", (0.5, 1))
def test_sliding_window_with_stride_1(fake_decoder, prediction_start):
    """Verifies that a sliding window with stride one is correct."""
    window_size = 2
    stride = 1

    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )
    x = torch.arange(5).reshape(5, 1)

    # note that the second timestep corresponds to the actual data
    # with window size 2 and centered prediction_start.
    expected = [
        [[0], [0]],
        [[0], [1]],
        [[1], [2]],
        [[2], [3]],
        [[3], [4]],
    ]
    actual = decoder._sliding_window(x)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("prediction_start", (0.5, 1))
def test_sliding_window_odd_window_size(fake_decoder, prediction_start):
    """Verifies sliding window works with odd window sizes."""
    x = torch.arange(5).reshape(5, 1)
    window_size = 3
    stride = 1
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )

    # notice that the middle timestep corresponds to the actual data
    expected = [
        [[0], [0], [1]],
        [[0], [1], [2]],
        [[1], [2], [3]],
        [[2], [3], [4]],
        [[3], [4], [0]],
    ]
    actual = decoder._sliding_window(x)
    np.testing.assert_array_equal(actual, expected)


def test_sliding_window_stride_100(fake_decoder):
    """Verifies sliding window is correct with stride > 1."""
    window_size = 200
    stride = 100
    prediction_start = 50
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )
    x = torch.arange(200).reshape(-1, 1)

    window_0 = torch.cat([torch.zeros(prediction_start), torch.arange(150)]).reshape(
        1, -1, 1
    )
    window_1 = torch.cat([torch.arange(50, 200), torch.zeros(50)]).reshape(1, -1, 1)
    expected = torch.cat([window_0, window_1], dim=0)
    actual = decoder._sliding_window(x)
    np.testing.assert_array_equal(actual, expected)


def test_sliding_window_stride_100_offset_start(fake_decoder):
    """Verifies sliding window is correct with a non-symmetric window start."""
    window_size = 200
    stride = 100
    prediction_start = 30
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )
    x = torch.arange(200).reshape(200, 1)
    window_0 = torch.cat([torch.zeros(30), torch.arange(170)]).reshape(1, 200, 1)
    window_1 = torch.cat([torch.arange(70, 200), torch.zeros(70)]).reshape(1, 200, 1)
    expected = torch.cat([window_0, window_1], dim=0)
    actual = decoder._sliding_window(x)
    np.testing.assert_array_equal(actual, expected)


def test_forward_pass_returns_correct_shape_with_perfect_overlap(fake_decoder, n_chars):
    """Verifies that strided forward pass returns correct shape.

    This test chooses strides, data sizes, and prediction start times that
    result in "perfect overlap" meaning the strided windows exactly cover
    the input data and there is no "extra" prediction due to padding etc.
    """
    window_size = 200
    stride = 100
    prediction_start = 50
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )
    x = torch.arange(200).reshape(1, 200, 1)

    actual = decoder(x)
    expected = torch.zeros(1, 200), torch.zeros((1, 200, n_chars))

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_forward_pass_returns_correct_shape_with_imperfect_overlap(
    fake_decoder, n_chars
):
    """Verifies that strided forward pass returns correct shape."""
    window_size = 200
    stride = 100
    prediction_start = 50
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )
    x = torch.arange(237).reshape(1, 237, 1)

    actual = decoder(x)
    expected = torch.zeros((1, 237)), torch.zeros((1, 237, n_chars))

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_decoder_handles_zero_channels(fake_decoder, n_chars):
    """Verify that the decoder does not error when no channels are present."""
    window_size = 2
    stride = 1
    prediction_start = 0.5
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
    )

    x = torch.arange(10).reshape(1, 10)  # (1,10) - no channels
    logits = decoder(x)
    actual = logits[0].shape, logits[1].shape
    expected = (1, 10), (1, 10, n_chars)

    assert actual == expected


def test_decoder_handles_extra_dimensions(fake_decoder, n_chars):
    """Verify the decoder can handle extra input dimensions.

    Note:
        The actual model is fake and doesn't care about extra
        dimensions - it returns the same data regardless of the
        input.  But before the model is used the SUT does do
        a sliding window of the input data so this tests taht
        the sliding window and the batching of that window
        is happening correctly.
    """
    window_size = 2
    stride = 1
    prediction_start = 0.5
    decoder = SlidingWindowDecoder(
        decoder=fake_decoder,
        window_size=window_size,
        stride=stride,
        prediction_start=prediction_start,
        batch_size=1,
    )

    x = torch.arange(10 * 2 * 3).reshape(1, 10, 2, 3)  # (1,10,2,3) - extra inputs
    logits = decoder(x)
    actual = logits[0].shape, logits[1].shape
    expected = (1, 10), (1, 10, n_chars)

    assert actual == expected
