r"""Efficient Transformer model for imagined handwriting decoding.


Reference:
    FNet: Mixing Tokens with Fourier Transforms: https://arxiv.org/abs/2105.03824 # noqa

Implementation inspired by:
    https://github.com/erksch/fnet-pytorch/blob/master/fnet.py
"""
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from imagined_handwriting.nn.utils import get_torch_function


class FNetEncoderLayer(nn.Module):
    """FNet encoder layer.

    This is meant to be a drop in replacement for nn.TransformerEncoderLayer,
    which can be used in a nn.TransformerEncoder.  As such we support the
    same forward signature but raise an error if any masks are passed to
    this layer.  FFT style attention does not support masking and failing
    silently could lead to some nasty bugs/bad conclusions, for example
    if the user things they are masking the future but they are not.

    Similarly, the initialization accepts `nhead` as an argument but
    for compatibility but it is unused.  In this case no error is
    raised if nhead>1 since this is a capacity parameter and we should
    expect correct outputs regardless of this value.


    Code is adapted from the original implementation:
    https://github.com/pytorch/pytorch/blob/56d1c755185c32cdc058cda190edfe139b895410/torch/nn/modules/transformer.py#L303 # noqa
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        """Initializes a FNetEncoderLayer object.

        d_model: the number of expected features in the input (required).
        nhead: sentinel value, unused by FNet but maintained for compatiblity with
            encoder layers.
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default="relu").
        layer_norm_eps: the eps value in layer normalization nn (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default:
            ``False`` (after).
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = batch_first
        self.nhead = nhead  # unused by FNet
        self.self_attn = FFTLayer()

        # implentation of the FeedForward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_torch_function(activation)
        self.activation = activation

    #     self.init_weights()

    # def init_weights(self):
    #     """Orthogonal initialization for the weights."""
    #     # for p in self.parameters():
    #     #     if p.dim() > 1:
    #     #         nn.init.orthogonal_(p)
    #     pass

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Runs a forward pass of the encoder layer.

        Args:
            src: The sequence to the encoder layer.
            src_mask: Not supported, if not None an error will be raised.
            src_key_padding_mask: Not supported, if not None and error will be raised.

        Raises:
            ValueErrror: If either of the mask tensors are not None.

        Returns:
            the output tensor of the encoder layer.
        """
        if src_mask is not None:
            raise ValueError(
                "FNetEncoderLayer does not support src_mask. "
                "If you need to use masking try multi-headed attention"
            )
        if src_key_padding_mask is not None:
            raise ValueError(
                "FNetEncoderLayer does not support src_key_padding_mask "
                "If you need to use masking try multi-headed attention"
            )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FFTLayer(nn.Module):
    """A fourier transform mixing layer.

    Reference:
        FNet: Mixing Tokens with Fourier Transforms, https://arxiv.org/abs/2105.03824 # noqa
    """

    def forward(self, x: Tensor) -> Tensor:
        """Runs a forward pass of the FFT layer.

        Args:
            x: A tensor of shape (batch_size, sequence_length, d_model).

        Raises:
            ValueError: If an attention mask is passed to the FFT layer.

        Returns:
            A tensor of shape (batch_size, sequence_length, d_model) encoded with
            FFT mixing.

        """
        return torch.fft.fft(torch.fft.fft(x.float(), dim=-1), dim=-2).real
