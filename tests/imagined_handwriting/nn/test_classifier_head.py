import torch

from imagined_handwriting.nn import ClassifierHead


def test_classifier_has_correct_weight_dimensions():
    """Test that the classifier head linear layers have the correct dimensions."""
    classifier = ClassifierHead(10, 5, d_hidden=20)
    assert classifier.hidden.weight.shape == (20, 10)  # stored backwards by torch
    assert classifier.out.weight.shape == (5, 20)  # stored backwards by torch


def test_classifier_head_returns_correct_shape():
    """Test that the classifier head returns the correct shape."""
    classifier = ClassifierHead(10, 5, d_hidden=20)
    x = torch.randn(3, 10)
    y = classifier(x)
    assert y.shape == (3, 5)
