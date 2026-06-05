"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
