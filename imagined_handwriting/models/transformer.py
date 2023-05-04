from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from imagined_handwriting.nn import (
    ChannelEmbedding,
    ClassifierHead,
    Downsample,
    FNetEncoderLayer,
    PositionalEncoding,
    Upsample,
)
from imagined_handwriting.settings import CHARACTERS


class HandwritingTransformer(nn.Module):
    """Transformer model for imagined handwriting decoding."""

    def __init__(
        self,
        *,
        input_channels: int = 192,
        num_classes: int = len(CHARACTERS.abbreviated) + 1,
        session_ids: Optional[List[str]] = None,
        attention: str = "self",
        d_model: int = 512,
        dim_feedforward: int = 2048,
        num_layers: int = 12,
        nhead: int = 8,
        encoder_dropout: float = 0.1,
        seq_dropout: float = 0.1,
        channel_dropout: float = 0.1,
        classifier_dropout: float = 0.1,
        activation: str = "gelu",
        positional_embedding: Optional[str] = "sinusoid",
        linear_embedding: bool = True,
        resample_factor: int = 5,
        resample_method: str = "mean",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        batch_first: bool = True,
        orthogonal_init: bool = False,
        **kwargs,
    ):
        """Initialize the HandwritingTransformer model.

        Args:
            input_channels: Number of input channels.
            num_classes: Number of output classes.
            session_ids: List of session ids.  If set, the model will have a
                separate embedding for each session.
            attention: Type of attention to use.  One of 'self', or 'fft'.
            d_model: Size of the intermediate representation.
            dim_feedforward: Size of the feedforward layer.
            num_layers: Number of layers in the encoder.
            nhead: Number of attention heads. Unused if attention is 'fnet'.
            encoder_dropout: Dropout rate for the encoder.
            seq_dropout: Dropout rate for the sequence dropout layer.
            channel_dropout: Dropout rate for the channel dropout layer.
            classifier_dropout: Dropout rate for the classifier layer.
            activation: Activation function to use.  One of 'relu', 'gelu'
                or any other function supported by torch.nn.functional.
            positional_embedding: Type of positional embedding to use.  One of
                'sinusoid' or 'learned'.
            linear_embedding: If True, the embedding layer will be a linear
                layer.  If False, it will be a standard token embedding layer.
            resample_factor: Factor to resample the output by. We first downsample
                the output before classification and then upsample it for final
                outputs.  This helps stabilize the decoding results.
            resample_method: Method to use for resampling.  One of 'mean' or
                'drop'. See `imagined_handwriting.nn.resample` for more details.
            layer_norm_eps: Epsilon value for layer normalization.
            norm_first: If True, the layer normalization will be applied before
                the residual connection.  If False, it will be applied after.
            batch_first: Must be True, batch_first=False is not supported.
        """

        super().__init__()
        if not batch_first:
            raise NotImplementedError("Only batch_first==True is supported")

        self.session_ids = session_ids

        self.input_channels = input_channels
        self.d_model = d_model
        self.linear_embedding = linear_embedding
        self.seq_dropout = seq_dropout
        self.channel_dropout = channel_dropout
        self.activation = activation

        self.embedding = self.make_embedding(
            self.session_ids,
            input_channels,
            d_model,
            linear=linear_embedding,
            seq_dropout=seq_dropout,
            channel_dropout=channel_dropout,
            activation=activation,
        )

        self.positional_embedding = PositionalEncoding(
            d_model=d_model,
            dropout=encoder_dropout,
            learned=positional_embedding == "learned",
        )

        self.encoder = self.make_encoder(
            attention=attention,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=encoder_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.downsample = Downsample(resample_factor, resample_method)
        self.classifier = ClassifierHead(
            d_model,
            num_classes,
            d_hidden=dim_feedforward,
            dropout=classifier_dropout,
            activation=activation,
        )
        self.upsample = Upsample(resample_factor)

        if orthogonal_init:
            self.init_weights()

    def init_weights(self):
        """Initialize the model weights.

        Orthogonal initialization is used

        See:
            Provable Benefit of Orthogonal Initialization in Optimizing Deep Linear Networks # noqa
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=1.0)

    def freeze_encoder(self):
        """Freeze the encoder and classifier weights."""
        for p in self.encoder.parameters():
            p.requires_grad = False

        for p in self.classifier.parameters():
            p.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("HandwritingTransformer")
        parser.add_argument("--attention", type=str, default="self")
        parser.add_argument("--d_model", type=int, default=512)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--num_layers", type=int, default=12)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--encoder_dropout", type=float, default=0.1)
        parser.add_argument("--seq_dropout", type=float, default=0.1)
        parser.add_argument("--channel_dropout", type=float, default=0.1)
        parser.add_argument("--classifier_dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="gelu")
        parser.add_argument("--positional_embedding", type=str, default="sinusoid")
        parser.add_argument("--linear_embedding", action="store_true")
        parser.add_argument("--resample_factor", type=int, default=5)
        parser.add_argument("--resample_method", type=str, default="mean")
        parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
        parser.add_argument("--norm_first", action="store_true")
        return parent_parser

    def forward(
        self, x: Tensor, session_id: Optional[str] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            session_id: If the model has multiple embeddings, this is the id of
                the embedding to use.

        Returns:
            logits_start, logits_char: Logits for the start of a character and
                the probability of each character.
        """
        embedding = self.get_embedding(session_id)
        x = embedding(x)
        x = self.positional_embedding(x)
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x[:, :, -1], x[:, :, :-1]

    def get_embedding(self, session_id: Optional[str] = None) -> nn.Module:
        """Get the embedding for a given session id."""
        if isinstance(session_id, List):
            assert all([id == session_id[0] for id in session_id])
            session_id = session_id[0]  # type: ignore

        if session_id is None and self.session_ids is None:
            return self.embedding

        if self.session_ids is None and session_id is not None:
            raise ValueError(
                "This model is trained with a single embedding, "
                "please do not specify a session_id in the forward "
                "pass."
            )

        if self.session_ids is not None and session_id is None:
            raise ValueError(
                "This model is trained with sessions "
                f"{self.session_ids}, please specify a session_id "
                "in the forward pass so the correct embedding layer "
                "is used."
            )

        if session_id not in self.session_ids:  # type: ignore
            raise ValueError(
                f"Session id {session_id} is not in the list of "
                f"session ids {self.session_ids}"
            )

        return self.embedding[self.replace_period(session_id)]  # type: ignore

    def make_embedding(
        self,
        session_ids: Optional[list],
        input_channels: int,
        d_model: int,
        *,
        linear: bool,
        seq_dropout: float,
        channel_dropout: float,
        activation: str,
    ):
        """Get the embedding layer according to the parameters.

        If session_ids is set then the embedding will be a module list of
        embeddings, one for each session.  This allows for different input layers
        for each session.
        """

        def _embedding():
            return ChannelEmbedding(
                input_channels,
                d_model,
                linear=linear,
                seq_dropout=seq_dropout,
                channel_dropout=channel_dropout,
                activation=activation,
            )

        if session_ids is None:
            return _embedding()

        return nn.ModuleDict(
            {self.replace_period(s): _embedding() for s in session_ids}
        )

    def make_encoder(
        self,
        *,
        attention,
        d_model,
        nhead,
        dim_feedforward,
        num_layers,
        dropout,
        activation,
        layer_norm_eps,
        batch_first,
        norm_first,
    ):
        """Makes the encoder for the transformer"""
        encoder_layer = get_encoder_layer(
            attention=attention,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def initialize_embedding(self, session_id, init_from):
        """Initialize the embedding for a session"""
        if session_id not in self.session_ids:
            raise ValueError(
                f"Can not initialize {session_id}, it does not exist in this model."
            )

        session_id = self.replace_period(session_id)
        init_from = self.replace_period(init_from)

        with torch.no_grad():
            self.embedding[session_id].load_state_dict(
                self.embedding[init_from].state_dict()
            )

    @staticmethod
    def replace_period(s: str):
        """Remove the period from a string"""
        return s.replace(".", "_")


def get_encoder_layer(attention: str, **kwargs) -> nn.Module:
    """Returns the encoder layer corresponding to the attention type.

    Args:
        attention: The type of attention to use.  For self-attention
            (standard transformer) use one of ['self', 'self-attention',
            'sa', 'learned'].
            For fft style attention (fnet) use one of ['fft', 'fnet']
        **kwargs: Additional arguments to pass to the encoder layer.
    """
    sa_options = ["self", "self-attention", "sa", "learned"]
    fft_options = ["fft", "fnet"]

    if attention in sa_options:
        return nn.TransformerEncoderLayer(**kwargs)
    elif attention in fft_options:
        config = {k: v for k, v in kwargs.items() if k != "nhead"}
        return FNetEncoderLayer(**config)
    else:
        raise ValueError(f"Unknown attention type: {attention}")
