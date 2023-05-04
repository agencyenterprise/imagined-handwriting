from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(logits: Tensor, labels: Tensor, loss_mask: Optional[Tensor]) -> Tensor:
    """Mean squared error loss.

    Computes mse_loss(sigmoid(logits), labels))

    Args:
        logits: A tensor of shape (batch_size, time) of raw
            outputs from the model.
        labels: A tensor of shape (batch_size,time) of ground truth
            character start times.
    Returns:
        The mean squared error loss over the batch.

    """
    pred = torch.sigmoid(logits)
    # return F.mse_loss(pred, labels)
    if loss_mask is not None:
        mse_loss = F.mse_loss(pred, labels, reduction="none")
        return torch.mean(mse_loss * loss_mask)
    return F.mse_loss(pred, labels)


def cross_entropy_loss(
    logits: Tensor, labels: Tensor, loss_mask: Optional[Tensor] = None
) -> Tensor:
    """Cross entropy loss.


    Args:
        logits: A tensor of shape (batch_size, time, characters) of
            raw output from the model.
        labels: A tensor of shape (batch_size, time) of dense labels
            or a tensor of shape (batch_size, time, characters) of one
            hot labels.  Prefer dense labels since pytorch has a more
            efficient implementation for dense labels.
        loss_mask: An optional tensor of shape (batch_size, time) whose
            i,j entry is 1 if loss should be considered at the j timestep
            of the ith batch, 0 otherwise.  If None, loss at every timestep
            is considered.  Default is None.

    Returns:
        The cross entropy loss over the batch.
    """
    # need to permute classes to dimension 1 according to pytorch docs
    logits = logits.permute(0, 2, 1)
    if labels.dim() == 3:
        # labels are 1 hot, permute classes to second dimension
        labels = labels.permute(0, 2, 1)

    if loss_mask is not None:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        return torch.mean(ce_loss * loss_mask)
    return F.cross_entropy(logits, labels)
