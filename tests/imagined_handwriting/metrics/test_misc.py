import torch

from imagined_handwriting.metrics.misc import frame_accuracy


def test_frame_accuracy_is_correct_without_loss_mask():
    """Verifies the correct accuracy is computed when *no* loss mask is provided"""

    logits = torch.tensor([[-100, 100], [-100, 100]]).reshape(1, 2, 2)
    labels = torch.tensor([[1, 0], [0, 1]]).reshape(1, 2, 2)

    acc = frame_accuracy(logits, labels)

    assert acc == 0.5


def test_frame_accuracy_is_correct_with_loss_mask():
    """Verifies the correct accuracy is computed when a loss mask is provided"""

    logits = torch.tensor([[-100, 100], [-100, 100]]).reshape(1, 2, 2)
    labels = torch.tensor([[1, 0], [0, 1]]).reshape(1, 2, 2)
    loss_mask = torch.tensor([[0, 1]])

    acc = frame_accuracy(logits, labels, loss_mask)
    assert acc == 1.0


def test_frame_accuracy_handles_one_hot_labels():
    """Verifies the correct accuracy is computed when one-hot labels are provided"""

    logits = torch.tensor([[-100, 100], [-100, 100]]).reshape(1, 2, 2)
    labels = torch.tensor([[0, 1]])
    acc = frame_accuracy(logits, labels)
    assert acc == 0.5
