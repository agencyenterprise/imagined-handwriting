import torch

from imagined_handwriting.nn.utils import get_torch_function


class ClassifierHead(torch.nn.Module):
    """A two layer classifier head for a transformer encoder."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        d_hidden: int,
        dropout: float = 0.0,
        activation="gelu",
    ):
        super().__init__()
        self.activation = get_torch_function(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden = torch.nn.Linear(in_features, d_hidden)
        self.out = torch.nn.Linear(d_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
