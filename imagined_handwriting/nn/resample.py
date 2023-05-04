import torch


class Downsample(torch.nn.Module):
    """Downsamples an input tensor along the time dimension."""

    def __init__(self, factor: int, method: str = "mean"):
        """Initializes the downsampling module.

        If the number of time steps is not divisible by factor than
        the last time steps are dropped.

        Args:
            factor: The downsampling factor.
            method: The downsampling method. One of ['mean', 'drop].
                If `mean`, the `factor` many time samples are averaged.
                If `drop` then we simply downsample by choosing every
                `factor` many time samples.

        """
        super().__init__()
        self.factor = factor
        if factor <= 0:
            raise ValueError("The downsampling factor must be positive.")
        if method not in ["mean", "drop"]:
            raise ValueError('method must be "mean" or "drop"')
        self.method = method

    def forward(self, x):
        """Downsamples the input tensor.

        Args:
            x: The input tensor of shape (batch, time, channels).

        Returns:
            The downsampled tensor of shape (batch, time // factor, channels).
        """
        if self.factor > x.size(1):
            raise ValueError(
                "The downsampling factor must be smaller than the number of time steps."
            )

        if self.method == "mean":
            B, T, H = x.shape  # batch, time, hidden
            f = self.factor
            num_frames = T - (T % f)
            x = x[:, :num_frames, :]
            return x.reshape(B, num_frames // f, f, H).mean(2)

        return x[:, :: self.factor, :]


class Upsample(torch.nn.Module):
    def __init__(self, factor: int):
        """Initializes the upsampling module.

        Args:
            factor: The upsampling factor.

        """
        super().__init__()
        if factor <= 0:
            raise ValueError("the up sampling factor must be positive")
        self.factor = factor

    def forward(self, x):
        return torch.repeat_interleave(x, self.factor, 1)
