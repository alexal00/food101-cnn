"""Grad-CAM utilities for visual model diagnostics."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn


@dataclass(frozen=True)
class GradCamOutput:
    """Grad-CAM result for one image and one target class."""

    class_index: int
    heatmap: np.ndarray
    logits: torch.Tensor


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer in a CNN-like model."""
    last_conv: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No nn.Conv2d layer was found for Grad-CAM.")
    return last_conv


def find_gradcam_target_layer(
    model: nn.Module,
    *,
    model_name: str | None = None,
    target_layer_name: str | None = None,
) -> nn.Module:
    """Resolve a Grad-CAM target layer with common backbone-aware defaults."""
    if target_layer_name is not None:
        return _resolve_named_layer(model, target_layer_name)

    named_modules = dict(model.named_modules())
    normalized_name = (model_name or "").lower()
    candidate_names: list[str] = []
    if normalized_name == "efficientnet_b0":
        candidate_names = ["features.8", "features.7"]
    elif normalized_name == "resnet50":
        candidate_names = ["layer4"]
    elif normalized_name == "convnext_tiny":
        candidate_names = ["features.7", "features.6"]

    for candidate_name in candidate_names:
        candidate = named_modules.get(candidate_name)
        if candidate is not None:
            return candidate

    return find_last_conv_layer(model)


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    *,
    class_index: int | None = None,
    target_layer: nn.Module | str | None = None,
    device: str | torch.device = "cpu",
) -> GradCamOutput:
    """Compute a normalized Grad-CAM heatmap for one image tensor."""
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    if image_tensor.ndim == 3:
        batch = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 4 and image_tensor.size(0) == 1:
        batch = image_tensor
    else:
        raise ValueError("image_tensor must have shape (C, H, W) or (1, C, H, W).")
    batch = batch.to(torch_device).requires_grad_(True)

    layer = _resolve_target_layer(model, target_layer)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(_module, _inputs, output) -> None:
        activations.append(output)

    def save_gradient(_module, _grad_inputs, grad_outputs) -> None:
        gradients.append(grad_outputs[0])

    forward_handle = layer.register_forward_hook(save_activation)
    backward_handle = layer.register_full_backward_hook(save_gradient)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        resolved_class_index = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
        score = logits[:, resolved_class_index].sum()
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = _normalize_heatmap(cam.squeeze(0).squeeze(0))
        return GradCamOutput(
            class_index=resolved_class_index,
            heatmap=heatmap.detach().cpu().numpy(),
            logits=logits.detach().cpu(),
        )
    finally:
        forward_handle.remove()
        backward_handle.remove()


def overlay_heatmap(
    image: Image.Image | str | Path,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
    cmap: str = "magma",
) -> np.ndarray:
    """Overlay a normalized heatmap on an RGB image."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    if isinstance(image, Image.Image):
        rgb_image = image.convert("RGB")
    else:
        rgb_image = Image.open(image).convert("RGB")

    image_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if heatmap_array.shape != image_array.shape[:2]:
        heatmap_tensor = torch.from_numpy(heatmap_array).unsqueeze(0).unsqueeze(0)
        heatmap_array = (
            F.interpolate(
                heatmap_tensor,
                size=image_array.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

    colored_heatmap = plt.get_cmap(cmap)(np.clip(heatmap_array, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * image_array + alpha * colored_heatmap, 0.0, 1.0)


def save_gradcam_overlay(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image: Image.Image | str | Path,
    output_path: str | Path,
    *,
    class_index: int | None = None,
    target_layer: nn.Module | str | None = None,
    model_name: str | None = None,
    device: str | torch.device = "cpu",
    cmap: str = "magma",
) -> Path | None:
    """Compute and save one Grad-CAM overlay, returning ``None`` on unsupported models."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target_layer is None:
            resolved_layer: nn.Module | str | None = find_gradcam_target_layer(
                model,
                model_name=model_name,
            )
        else:
            resolved_layer = target_layer
        result = compute_gradcam(
            model,
            image_tensor,
            class_index=class_index,
            target_layer=resolved_layer,
            device=device,
        )
        overlay = overlay_heatmap(image, result.heatmap, cmap=cmap)
    except Exception as exc:
        warnings.warn(f"Grad-CAM skipped for {path.name}: {exc}", RuntimeWarning, stacklevel=2)
        return None

    plt.imsave(path, overlay)
    return path


def save_gradcam_sets_from_predictions(
    model: nn.Module,
    predictions: pd.DataFrame,
    *,
    class_names: list[str],
    transform,
    output_dir: str | Path = "outputs/figures/gradcam",
    project_root: str | Path | None = None,
    model_name: str | None = None,
    target_layer_name: str | None = None,
    device: str | torch.device = "cpu",
    examples_per_set: int = 3,
) -> dict[str, list[Path]]:
    """Save small Grad-CAM sets for correct, wrong, and ambiguous predictions."""
    from food101_cnn.evaluation.errors import normalize_prediction_dataframe, resolve_image_path

    frame = normalize_prediction_dataframe(predictions)
    sets = {
        "correct_high_confidence": frame[frame["is_correct"]].sort_values(
            "top1_confidence",
            ascending=False,
        ),
        "wrong_high_confidence": frame[~frame["is_correct"]].sort_values(
            "top1_confidence",
            ascending=False,
        ),
        "ambiguous_low_confidence": frame.sort_values("top1_confidence", ascending=True),
    }

    root = Path(output_dir).expanduser().resolve()
    outputs: dict[str, list[Path]] = {name: [] for name in sets}
    target_layer = target_layer_name
    for set_name, set_frame in sets.items():
        set_dir = root / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for row in set_frame.itertuples(index=False):
            if saved >= examples_per_set:
                break
            image_path = resolve_image_path(str(row.image_path), project_root=project_root)
            if image_path is None:
                continue
            pred_index = _prediction_index(row, class_names)
            if pred_index is None:
                continue
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            output_path = set_dir / (
                f"{saved:02d}_true-{_safe_name(row.true_class)}"
                f"_pred-{_safe_name(row.pred_class)}.png"
            )
            saved_path = save_gradcam_overlay(
                model,
                image_tensor,
                image,
                output_path,
                class_index=pred_index,
                target_layer=target_layer,
                model_name=model_name,
                device=device,
            )
            if saved_path is not None:
                outputs[set_name].append(saved_path)
                saved += 1
    return outputs


def _resolve_target_layer(model: nn.Module, target_layer: nn.Module | str | None) -> nn.Module:
    if target_layer is None:
        return find_gradcam_target_layer(model)
    if isinstance(target_layer, str):
        return _resolve_named_layer(model, target_layer)
    return target_layer


def _resolve_named_layer(model: nn.Module, target_layer_name: str) -> nn.Module:
    named_modules = dict(model.named_modules())
    if target_layer_name not in named_modules:
        raise ValueError(f"Target layer not found: {target_layer_name}")
    return named_modules[target_layer_name]


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    minimum = heatmap.min()
    maximum = heatmap.max()
    if torch.isclose(maximum, minimum):
        return torch.zeros_like(heatmap)
    return (heatmap - minimum) / (maximum - minimum)


def _prediction_index(row, class_names: list[str]) -> int | None:
    if hasattr(row, "pred_idx"):
        try:
            return int(row.pred_idx)
        except (TypeError, ValueError):
            pass
    pred_class = str(row.pred_class)
    if pred_class not in class_names:
        return None
    return class_names.index(pred_class)


def _safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")
