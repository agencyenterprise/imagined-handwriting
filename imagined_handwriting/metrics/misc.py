from typing import Optional

import torch
from torch import Tensor


def frame_accuracy(
    logits: Tensor, labels: Tensor, loss_mask: Optional[Tensor] = None
) -> Tensor:
    """Computes average accuracy computed over every time step.

    Args:
        logits: A tensor of shape (batch_size, time, characters) of
            raw output from the model.
        labels: A dense tensor of shape (batch_size, time) whose i,j entry
            give the character index at time j in batch i or a one hot
            tensor of shape (batch_size, time, characters) of one hot labels.
        loss_mask: An optional tensor of shape (batch_size, time) whose
            i,j entry is 1 if the the j time step of the ith batch should be
            included in the average accuracy, 0 otherwise. If None,
            all time steps will be used in the average. Default is None.

    Returns:
        The average accuracy taken over the each time step.

    """

    pred = torch.argmax(logits, dim=-1)
    if labels.dim() == 3:
        correct = torch.argmax(labels, dim=-1)
    else:
        correct = labels
    if loss_mask is not None:
        acc = torch.sum(loss_mask * torch.eq(pred, correct)) / torch.sum(loss_mask)
    else:
        acc = torch.mean(torch.eq(pred, correct).float())
    return acc
