"""Model registry for configuration-driven construction."""

from collections.abc import Mapping
from typing import Any

from torch import nn

from food101_cnn.models.baseline_cnn import build_baseline_cnn, build_local_baseline_cnn
from food101_cnn.models.transfer import (
    SUPPORTED_TRANSFER_MODELS,
    build_transfer_model,
    freeze_backbone_parameters,
    unfreeze_all_parameters,
    unfreeze_last_stages,
)

SUPPORTED_MODELS = (
    "baseline_cnn",
    "baseline_cnn_local",
    "baseline_cnn_simple",
    *SUPPORTED_TRANSFER_MODELS,
)


def build_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.3,
    freeze_backbone_initially: bool = False,
) -> nn.Module:
    """Build a model by registry name."""
    normalized_name = model_name.lower()
    if normalized_name == "baseline_cnn":
        if pretrained:
            raise ValueError("baseline_cnn does not support pretrained weights.")
        return build_baseline_cnn(num_classes=num_classes, dropout=dropout)
    if normalized_name in {"baseline_cnn_local", "baseline_cnn_simple"}:
        if pretrained:
            raise ValueError(f"{normalized_name} does not support pretrained weights.")
        return build_local_baseline_cnn(num_classes=num_classes, dropout=dropout)

    if normalized_name in SUPPORTED_TRANSFER_MODELS:
        return build_transfer_model(
            normalized_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone_initially,
        )

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")


def build_model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Build a model from a loaded YAML config mapping."""
    model_config = config["model"]
    return build_model(
        str(model_config["name"]),
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config.get("pretrained", False)),
        dropout=float(model_config.get("dropout", 0.3)),
        freeze_backbone_initially=bool(
            model_config.get("freeze_backbone_initially", False)
        ),
    )


__all__ = [
    "SUPPORTED_MODELS",
    "build_model",
    "build_model_from_config",
    "freeze_backbone_parameters",
    "unfreeze_all_parameters",
    "unfreeze_last_stages",
]
