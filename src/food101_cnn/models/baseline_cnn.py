"""Custom CNN baseline for Food-101 classification."""

import torch
from torch import nn


class Food101BaselineCNN(nn.Module):
    """Small CNN baseline built from scratch for pedagogical comparison."""

    def __init__(self, num_classes: int = 101, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.features = nn.Sequential(
            _conv_block(3, 32, pool=True),
            _conv_block(32, 64, pool=True),
            _conv_block(64, 128, pool=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        features = self.features(inputs)
        return self.classifier(features)


class Food101LocalBaselineCNN(nn.Module):
    """Simpler CNN baseline for local Mac/CPU/MPS experiments."""

    def __init__(self, num_classes: int = 101, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.features = nn.Sequential(
            _simple_conv_block(3, 16),
            _simple_conv_block(16, 32),
            _simple_conv_block(32, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        features = self.features(inputs)
        return self.classifier(features)


def build_baseline_cnn(num_classes: int = 101, dropout: float = 0.3) -> Food101BaselineCNN:
    """Build the custom CNN baseline."""
    return Food101BaselineCNN(num_classes=num_classes, dropout=dropout)


def build_local_baseline_cnn(
    num_classes: int = 101,
    dropout: float = 0.3,
) -> Food101LocalBaselineCNN:
    """Build the simpler local CNN baseline."""
    return Food101LocalBaselineCNN(num_classes=num_classes, dropout=dropout)


def _conv_block(in_channels: int, out_channels: int, *, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*layers)


def _simple_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )
