import torch
from torch import Tensor, nn


class DimensionDropout(nn.Module):
    """Dropout of all units along a single dimension.

    For example if applied to a tensor of shape (batch_size, seq_len, channels)
    then we have

    dim=1:
        A random number of sequence steps will be chosen and all channels
        at that that step will be set to zero.
    dim=2:
        A random number of channels will be chosen and all sequence steps
        at that channel will be set to zero.

    Note that this implicitly assumes the first dimension is the batch.
    """

    def __init__(self, p: float, dim: int):
        """Initializes DimensionDropout.

        Args:
            p: The percentage of units to dropout.
            dim: The dimension to dropout.
        """
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the dropout layer.

        Args:
            x: An input tensor of shape (batch_size, ...).

        Returns:
            The input with a random number of elements along a specified
            dimension set to zero.
        """
        if self.training:
            shape = [x.shape[0]] + [1 for _ in range(x.ndim - 1)]
            shape[self.dim] = x.shape[self.dim]
            mask = torch.empty(*shape).bernoulli_(1 - self.p).type_as(x).bool()
            x = x * mask
            x = x.div(1 / (1 - self.p))
        return x


class SequenceDropout(DimensionDropout):
    """A dropout layer along the sequence (1st) dimension.

    Expects inputs of shape (batch_size, seq_len, ...).
    """

    def __init__(self, p: float):
        """Initializes a SequenceDropout.

        Args:
            p: The percentage of units to dropout.
        """
        super().__init__(p, dim=1)


class ChannelDropout(DimensionDropout):
    """A channel dropout layer.

    Expects inputs of shape (batch_size, seq_len, channels, ...).

    """

    def __init__(self, p: float):
        """Initializes a ChannelDropout.

        Args:
            p: The percentage of units to dropout.
        """
        super().__init__(p, dim=2)
