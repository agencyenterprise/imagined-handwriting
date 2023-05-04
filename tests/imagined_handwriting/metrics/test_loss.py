import torch

from imagined_handwriting.metrics.loss import cross_entropy_loss


def test_cross_entropy_loss_handles_sparse_labels():
    """Verifies that sparse labels are handled correctly."""
    logits = torch.tensor([-1000.0, 1000.0]).reshape(1, 1, 2)
    labels = torch.tensor([1]).reshape(1, 1)
    loss = cross_entropy_loss(logits, labels)
    assert 0.0 <= loss.item() < 1e-6


def test_cross_entropy_loss_handles_onehot_labels():
    """Verifies that dense one hot labels are handled correctly."""
    logits = torch.tensor([-1000.0, 1000.0]).reshape(1, 1, 2)
    labels = torch.tensor([0, 1], dtype=torch.float32).reshape(1, 1, 2)
    loss = cross_entropy_loss(logits, labels)
    assert 0.0 <= loss.item() < 1e-6
