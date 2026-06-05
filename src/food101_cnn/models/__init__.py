"""Model definitions and factories."""

from food101_cnn.models.baseline_cnn import (
    Food101BaselineCNN,
    Food101LocalBaselineCNN,
    build_baseline_cnn,
    build_local_baseline_cnn,
)
from food101_cnn.models.registry import (
    SUPPORTED_MODELS,
    build_model,
    build_model_from_config,
)
from food101_cnn.models.transfer import (
    build_transfer_model,
    freeze_backbone_parameters,
    unfreeze_all_parameters,
    unfreeze_last_stages,
)

__all__ = [
    "Food101BaselineCNN",
    "Food101LocalBaselineCNN",
    "SUPPORTED_MODELS",
    "build_baseline_cnn",
    "build_local_baseline_cnn",
    "build_model",
    "build_model_from_config",
    "build_transfer_model",
    "freeze_backbone_parameters",
    "unfreeze_all_parameters",
    "unfreeze_last_stages",
]
