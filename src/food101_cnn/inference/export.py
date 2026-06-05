"""Model export helpers."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


def export_pytorch_state(
    model: nn.Module,
    output_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Export a native PyTorch state-dict package."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "class_names": class_names,
            "extra_state": extra_state or {},
        },
        path,
    )
    return path


def export_onnx_model(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device | str = "cpu",
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a model to ONNX when the local PyTorch install supports it."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    dummy_input = torch.randn(*input_shape, device=torch_device)
    dynamic_axes = (
        {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    return path


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Load a checkpoint or state_dict into a model."""
    checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a state_dict or contain model_state_dict.")
    model.load_state_dict(state_dict)
    return model
