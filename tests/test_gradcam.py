import numpy as np
import torch
from PIL import Image
from torch import nn

from food101_cnn.interpretability.gradcam import (
    compute_gradcam,
    find_last_conv_layer,
    overlay_heatmap,
)


def test_compute_gradcam_returns_normalized_heatmap() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    )

    result = compute_gradcam(model, torch.rand(3, 16, 16), class_index=1)

    assert result.class_index == 1
    assert result.heatmap.shape == (16, 16)
    assert np.isfinite(result.heatmap).all()
    assert 0.0 <= float(result.heatmap.min()) <= float(result.heatmap.max()) <= 1.0


def test_find_last_conv_layer_and_overlay_heatmap() -> None:
    model = nn.Sequential(nn.Conv2d(3, 2, kernel_size=1), nn.Conv2d(2, 1, kernel_size=1))
    image = Image.new("RGB", (12, 8), color=(120, 80, 40))
    heatmap = np.ones((8, 12), dtype=np.float32)

    assert find_last_conv_layer(model) is model[-1]
    overlay = overlay_heatmap(image, heatmap)
    assert overlay.shape == (8, 12, 3)
    assert np.isfinite(overlay).all()
