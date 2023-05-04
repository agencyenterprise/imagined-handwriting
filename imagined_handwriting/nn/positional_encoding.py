import math
from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the input tensor."""

    def __init__(
        self,
        *,
        d_model: int,
        dropout: float = 0.1,
        max_len: Optional[int] = None,
        learned: bool = False,
    ):
        """Initializes a PositionalEncoding object.

        Args:
            d_model: the hidden dimension of encoding.
            dropout: the dropout rate.
            max_len: the maximum length of the input tensor.
            learned: whether to use learned positional encoding.
        """
        super().__init__()
        if max_len is None:
            max_len = 500 if learned else 5000
        self.dropout = nn.Dropout(p=dropout)
        if learned:
            self.encoder = LearnedPositionalEncoding(d_model, dropout, max_len)
        else:
            self.encoder = SinusoidEncoding(d_model, dropout, max_len)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class SinusoidEncoding(nn.Module):
    """Relative sinusodial positional encoding.

    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    in particular the only real change is forcing batch_first=True by
    re-arranging the indexing.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initializes a PositionalEncoding object."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model + d_model % 2)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :, :d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input tensor and applies dropout.

        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]

        Returns:
            The input tensor with positional encoding added and dropout.
        """
        # mypy doesn't know that pe is a Tensor
        assert isinstance(self.pe, Tensor)
        if x.size(1) > self.pe.size(1):
            raise RuntimeError(
                f"Input tensor has length {x.size(1)}, but "
                f"maximum length is {self.pe.size(1)}."
            )
        x = x + self.pe[0, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding.

    Instead of encoding position with fixed encoding,
    we learn a positional encoding with an embedding matrix.  Note that
    for long sequence lengths, this matrix becomes very large.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        """Initializes a LearnedPositionalEncoding object."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        self.position_id: torch.Tensor
        self.register_buffer("position_id", torch.arange(max_len))

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input and applies dropout.

        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)

        Returns:
            The input tensor summed with added encoding and dropout.
        """
        if x.size(1) > self.position_id.size(0):
            raise RuntimeError(
                f"Input tensor has length {x.size(1)}, but "
                f"maximum length is {self.position_id.size(0)}."
            )
        ids = self.position_id[: x.size(1)]  # type: ignore
        x = x + self.embedding(ids)
        return self.dropout(x)
