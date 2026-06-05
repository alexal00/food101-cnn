"""Transfer-learning model factory and freezing utilities."""

from collections.abc import Iterable
from typing import Literal

from torch import nn
from torchvision import models

TransferModelName = Literal["efficientnet_b0", "resnet50", "convnext_tiny"]
SUPPORTED_TRANSFER_MODELS = ("efficientnet_b0", "resnet50", "convnext_tiny")


def build_transfer_model(
    model_name: str,
    *,
    num_classes: int = 101,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a torchvision transfer-learning model with a Food-101 head."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1).")

    normalized_name = model_name.lower()
    if normalized_name == "efficientnet_b0":
        model = _build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif normalized_name == "resnet50":
        model = _build_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif normalized_name == "convnext_tiny":
        model = _build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        supported = ", ".join(SUPPORTED_TRANSFER_MODELS)
        raise ValueError(f"Unsupported transfer model '{model_name}'. Supported: {supported}")

    if freeze_backbone:
        freeze_backbone_parameters(model)

    return model


def freeze_backbone_parameters(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in classifier_parameters(model):
        parameter.requires_grad = True


def unfreeze_all_parameters(model: nn.Module) -> None:
    """Unfreeze every trainable parameter in a model."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def unfreeze_last_stages(model: nn.Module, num_stages: int) -> None:
    """Unfreeze the classifier and the last top-level backbone stages."""
    if num_stages < 0:
        raise ValueError("num_stages must be non-negative.")

    freeze_backbone_parameters(model)
    if num_stages == 0:
        return

    backbone_children = _backbone_stages(model)
    for module in backbone_children[-num_stages:]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def classifier_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield classifier-head parameters for supported model layouts."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        yield from model.classifier.parameters()
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        yield from model.fc.parameters()
    else:
        raise ValueError("Model does not expose a supported classifier head.")


def _build_efficientnet_b0(
    *,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = _classifier_head(in_features, num_classes, dropout)
    return model


def _build_resnet50(
    *,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = _classifier_head(in_features, num_classes, dropout)
    return model


def _build_convnext_tiny(
    *,
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    final_linear = model.classifier[-1]
    if not isinstance(final_linear, nn.Linear):
        raise TypeError("Expected ConvNeXt classifier to end with nn.Linear.")

    classifier_prefix = list(model.classifier.children())[:-1]
    head_layers = [*classifier_prefix]
    if dropout > 0:
        head_layers.append(nn.Dropout(p=dropout))
    head_layers.append(nn.Linear(final_linear.in_features, num_classes))
    model.classifier = nn.Sequential(*head_layers)
    return model


def _classifier_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def _backbone_stages(model: nn.Module) -> list[nn.Module]:
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        feature_stages = [
            module for module in model.features.children() if _has_parameters(module)
        ]
        if feature_stages:
            return feature_stages
    return _top_level_backbone_children(model)


def _top_level_backbone_children(model: nn.Module) -> list[nn.Module]:
    excluded = {"classifier", "fc"}
    return [
        module
        for name, module in model.named_children()
        if name not in excluded and _has_parameters(module)
    ]


def _has_parameters(module: nn.Module) -> bool:
    return next(module.parameters(), None) is not None
