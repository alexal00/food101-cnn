"""Loss construction helpers."""

import torch
from torch import nn


def build_classification_loss(label_smoothing: float = 0.0) -> nn.Module:
    """Build cross-entropy loss with optional label smoothing."""
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in the range [0, 1).")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
