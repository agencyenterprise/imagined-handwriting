from typing import Tuple, Union

import numpy as np
import torch


class SlidingWindowDecoder(torch.nn.Module):
    """Performs handwriting decoding via a sliding window.

    This is intended to be used with models that require a fixed
    length window of input data, e.g. a transformer model. This
    decoder will wrap such a model and handle decoding arbitrary
    length inputs.  This is convenience for offline decoding of
    complete sentences (e.g. during validation or testing).

    The input is batched into overlapping windows of size
    `window_size` and the model is run on each window. The
    arguments `stride` and `prediction_start` control how
    much overlap there is between windows and where the
    predictions are made within each window.

    Example:
        For a model that accepts windows of size 200, setting window_size=200,
        stride=100, and prediction_start=50 the following schematic shows how
        the predictions are made.

    |---- past context ---- | ---- prediction  ---- | ---- future context ---- |
    |        50             |        100            |           50             |

        The windows will then be shifted by stride==100 and the same prediction
        scheme will be used.  Note that this means that every single time step
        gets a single prediction but that past and future context can overlap
        with previous windows.

    The sliding window decoder automatically handles padding so that the the
    predictions are exactly aligned with the input logits.


    Note:
        If you have a model that is distributed over multiple GPUs this will
        fail.  We use the GPU of the first model parameter to determine where
        to map that data.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        window_size: int,
        stride: int = 1,
        prediction_start: Union[float, int] = 0.5,
        batch_size: int = 64,
    ):
        """Initializes a SlidingWindowDecoder.

        Args:
            decoder: A pytorch model.
            window_size: The size of the sliding window. This must be the same
                as the sequence length accepted by the decoder model.
            stride: The number of timesteps to move by when sliding to the
                next window. Default is 1.
            prediction_start: The location within the sliding window to
                start extracting logits from.  If an integer it is treated
                as an index in [0, window_size) for the relative starting
                position for each sliding window.  If a float it is consider
                as a percentage in [0,1] and the integer index start location
                will be computed as
                    `int(np.floor(prediction_start * window_size))`
            batch_size: The number of windows to send to the model at a single
                time. If you are running into OOM errors on GPU use a smaller
                batch size.
        """
        super().__init__()
        self.decoder = decoder
        self.window_size = window_size
        self.stride = stride
        self._prediction_start = prediction_start
        self.batch_size = batch_size
        self.prediction_start = self._coerce_to_integer_index(prediction_start)
        self.prediction_end = self.prediction_start + stride
        self.device = next(self.decoder.parameters()).device

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass of the model against the input array.

        Note:
            The underlying decoder model is put in evaluation mode
            and gradients are not accumulated during this forward
            pass.

        Args:
            x: The input array of shape (batch, timesteps, ...)
                to decode.
            **kwargs: Additional keyword arguments to pass to the
                the wrapped model's forward pass.  Used when we need
                to pass in a session id to identify the correct input
                layer.

        Returns:
            logits for start times and characters of shape
            (batch, timesteps) and (batch, timesteps, characters)

        """
        l_start = []
        l_char = []
        timesteps = x.shape[1]
        for example in x:
            with torch.no_grad():
                logits_start, logits_char = self._forward_single_example(
                    example, timesteps, **kwargs
                )
            l_start.append(logits_start)
            l_char.append(logits_char)

        logits_start = torch.stack(l_start, dim=0)
        logits_char = torch.stack(l_char, dim=0)

        return logits_start, logits_char

    def _forward_single_example(
        self, x: torch.Tensor, timesteps: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a forward pass for a single sentence."""
        logits_start, logits_char = self._sliding_window_logits(x, **kwargs)
        logits_start = self._trim_logits(logits_start, timesteps)
        logits_char = self._trim_logits(logits_char, timesteps)
        return logits_start, logits_char

    def _sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        """Converts input array into a sliding window view of the array.

        Args:
            x: A tensor of shape (time,channels) that represents the neural
                activity for a single sentence.

        Returns:
            An array of shape (windows, window_size, channels) which is a view
            of the original array (with some windows possible padded).

        Note:
            * This function pads the array so that the number of windows is
            equal to the original number of timesteps.  This is similar to
            using the padding="same" option on convolutions from tensorflow
            and pytorch.


        Example:
            >>> self.window_size = 100
            >>> self.alignment = 30
            Then the first window will be a view of 30 padded points and 70
            time points from the array.

            >>> self.window_size = 100
            >>> self.alignment = .9
            Then the first window will be a view of 90 padded points and
            10 time points from the array.
        """
        # convert to numpy for sliding window
        x = x.cpu().numpy()
        timesteps = x.shape[0]
        pad = self._get_padding(timesteps)
        padding = [(0, 0) for _ in range(x.ndim)]
        padding[0] = pad
        x_pad = np.pad(x, padding)
        x_slide = np.lib.stride_tricks.sliding_window_view(
            x_pad, self.window_size, axis=0
        )
        x_slide = np.swapaxes(x_slide, -1, 1)
        window = x_slide[: x.shape[0]][:: self.stride]

        # convert back to tensor - note still on cpu, we will
        # change this later when making predictions
        return torch.tensor(window)

    def _sliding_window_logits(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the logits for each of the sliding windows."""
        start, end = self.prediction_start, self.prediction_end
        x_window = self._sliding_window(x)
        dataloader = self._batchify(x_window)
        logits_start = []
        logits_char = []
        for batch in dataloader:
            batch = batch.to(self.device)
            _logits_start, _logits_char = self.decoder(batch, **kwargs)

            logits_start.append(_logits_start[:, start:end])
            logits_char.append(_logits_char[:, start:end])

        return torch.cat(logits_start), torch.cat(logits_char)

    def _batchify(self, windows: torch.Tensor) -> torch.utils.data.DataLoader:
        """Creates a dataloader from a sliding window array view.

        The sliding window view can have many windows. In the worst case
        of stride 1 the number of windows will be approximately equal to the
        number of timebins in the sentences (1000s) so we need to break up
        the windows into batches otherwise we will run into OOM errors on
        our GPUs.
        """
        dataset = TensorDataset(windows)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, drop_last=False
        )

    def _trim_logits(self, logits: torch.Tensor, timesteps: int) -> torch.Tensor:
        """Trims the logits so there is one prediction per input timestep.

        Becuse of padding, the last window may contain predictions for
        timesteps beyond the end of the sentence.  This function trims any
        "extra" padding predictions so that there are just as many logits
        as there are original timesteps.

        Args:
            logits: The logits to trim of shape (windows, window_size,...)
            timesteps: The number of timesteps in the original sentence.

        Returns:
            The logits collapse to (timesteps, ...)
        """
        windows, pred_size = logits.shape[:2]
        logit_timesteps = windows * pred_size
        n_extra = logit_timesteps - timesteps

        def _trim(logits):
            return logits[: pred_size - n_extra]

        if n_extra == 0:
            # predictions perfectly overlap, a miracle or a test case :)
            return logits.reshape((-1,) + logits.shape[2:])

        if logits.shape[0] == 1:
            # only one window, trim and return
            return _trim(logits[0])

        logits_batch = logits[:-1]  # (windows-1, window_size, ...)
        logits_last = _trim(logits[-1])
        logits_batch = logits_batch.reshape((-1,) + logits_batch.shape[2:])

        return torch.cat([logits_batch, logits_last], dim=0)

    def _get_padding(self, timesteps):
        """Gets the correct front and back padding for sliding window.

        The back padding is calculated by sliding the window enough times
        so that prediction portion "covers" all the timesteps.
        """
        # since we add pred_start padding at the front
        first_end = self.prediction_end - self.prediction_start
        # enough to cover timesteps with predictions
        n_strides = int(np.ceil((timesteps - first_end) / self.stride))
        # need enough padding for whole window, not just predictions
        pad_end = (
            self.window_size
            + n_strides * self.stride
            - self.prediction_start
            - timesteps
        )
        return self.prediction_start, pad_end

    def _coerce_to_integer_index(self, x: Union[float, int]):
        """Coerces the start time to an integer index."""
        if isinstance(x, float):
            if x < 0 or x > 1:
                raise ValueError(f"When x is a float it must be in [0,1] but given {x}")
            return int(np.floor(x * self.window_size))
        elif isinstance(x, int):
            if x < 0 or x >= self.window_size:
                raise ValueError(
                    "When x is an int it must be in [0, window_size) "
                    f" but given {x} with window size {self.window_size}"
                )
            return x
        else:
            raise ValueError(f"Expected float or int but got {type(x)}")


class TensorDataset(torch.utils.data.Dataset):
    """A simple pytorch dataset wrapping a numpy array."""

    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]
