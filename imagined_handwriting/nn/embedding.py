from torch import Tensor, nn

from imagined_handwriting.nn import ChannelDropout, SequenceDropout
from imagined_handwriting.nn.utils import get_torch_function


class ChannelEmbedding(nn.Module):
    """Maps channels to a new vector space either by a linear map or
    an embedding lookup.

    The raw channel data shared by Willet et al. is binned threshold
    crossings.  These binned threshold crossing are integer counts which
    can be thought of as per-channel tokens. We can use standard embedding
    techniques to embed these tokens.  Alternatively, if we normalize
    the neural data (e.g z-scoring by blocks as done in Willet et al.) then
    the inputs are no longer counts but instead are real valued. In this
    case our embedding is a linear map to a new vector space.  This is
    useful since we can control the dimension of the model this way and
    we can also instantiate multiple linear embeddings per session.

    Important:
        Care must be taken when using this module with standard token embedding.
        If we use the counts as tokens then the embedding matrix will be size
        `[max_count, embedding_dim]` and each channel will get a shared representation
        of the embedded spike count.  Instead if we want per-channel embeddings we
        need to first manipulate the data so that channel i has tokens in the range
        `[i*max_count, (i+1)*max_count]` and then instantiate this class with
        `num_channels = num_channels*max_count`.

    Important:
        If using token count embeddings we will end up with a tensor of shape
            `[batch_size, time_steps, embedding_dim*num_channels]`.
        To reduce the dimensionality of this tensor back to
            `[batch_size, time_steps, embedding_dim]`
        we can use a linear projection layer. This is enabled by default but
        can be disabled by setting `projection=False`.

    """

    def __init__(
        self,
        num_channels: int,
        embedding_dim: int,
        linear=True,
        projection=True,
        seq_dropout: float = 0.0,
        channel_dropout: float = 0.0,
        activation: str = "gelu",
        **kwargs
    ):
        """Initialize a channel embedding"""
        super().__init__()
        self.linear = linear
        self.projection = projection
        self.embedding: nn.Module
        self.seq_dropout = SequenceDropout(seq_dropout)
        self.channel_dropout = ChannelDropout(channel_dropout)
        self.activation = get_torch_function(activation)
        if linear:
            self.embedding = nn.Linear(num_channels, embedding_dim, **kwargs)
        else:
            self.embedding = nn.Embedding(num_channels, embedding_dim, **kwargs)
            if projection:
                self.proj = nn.Linear(
                    embedding_dim * num_channels, embedding_dim, bias=False
                )

    def forward(self, x: Tensor) -> Tensor:
        """Embed the channels of x"""
        x = self.embedding(x)
        if not self.linear:
            x = x.reshape(x.shape[0], x.shape[1], -1)  # stack channel embeddings
            x = self.seq_dropout(x)
            x = self.channel_dropout(x)
            x = self.activation(x)
            if self.projection:
                x = self.proj(x.reshape(x.shape[0], x.shape[1], -1))
        else:
            x = self.seq_dropout(x)
            x = self.channel_dropout(x)
        return x
