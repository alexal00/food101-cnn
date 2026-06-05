import pytest
import torch

from food101_cnn.config import load_config
from food101_cnn.models.registry import build_model, build_model_from_config
from food101_cnn.models.transfer import (
    build_transfer_model,
    freeze_backbone_parameters,
    unfreeze_all_parameters,
    unfreeze_last_stages,
)


def test_baseline_cnn_forward_shape() -> None:
    model = build_model(
        "baseline_cnn",
        num_classes=7,
        pretrained=False,
        dropout=0.2,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 7)


def test_local_baseline_cnn_forward_shape_and_smaller_capacity() -> None:
    baseline = build_model(
        "baseline_cnn",
        num_classes=7,
        pretrained=False,
        dropout=0.2,
    )
    local = build_model(
        "baseline_cnn_local",
        num_classes=7,
        pretrained=False,
        dropout=0.2,
    )
    local.eval()

    with torch.no_grad():
        logits = local(torch.randn(2, 3, 64, 64))

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    local_params = sum(parameter.numel() for parameter in local.parameters())
    assert logits.shape == (2, 7)
    assert local_params < baseline_params


@pytest.mark.parametrize("model_name", ["efficientnet_b0", "resnet50", "convnext_tiny"])
def test_transfer_model_forward_shape_without_pretrained_weights(model_name: str) -> None:
    model = build_transfer_model(
        model_name,
        num_classes=11,
        pretrained=False,
        dropout=0.1,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 11)


def test_build_model_from_config_uses_yaml_model_fields() -> None:
    config = load_config("configs/baseline_cnn.yaml")
    config["model"]["num_classes"] = 5

    model = build_model_from_config(config)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 5)


def test_build_model_from_local_config_uses_simple_model() -> None:
    config = load_config("configs/baseline_cnn_local.yaml")
    config["model"]["num_classes"] = 5

    model = build_model_from_config(config)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 5)


def test_build_model_from_simple_config_uses_lightweight_model() -> None:
    config = load_config("configs/baseline_cnn_simple.yaml")
    config["model"]["num_classes"] = 5

    baseline = build_model("baseline_cnn", num_classes=5, pretrained=False)
    model = build_model_from_config(config)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    simple_params = sum(parameter.numel() for parameter in model.parameters())
    assert logits.shape == (1, 5)
    assert simple_params < baseline_params


def test_freeze_backbone_keeps_classifier_trainable() -> None:
    model = build_transfer_model(
        "efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )

    classifier_params = list(model.classifier.parameters())
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.")
    ]

    assert classifier_params
    assert all(parameter.requires_grad for parameter in classifier_params)
    assert backbone_params
    assert not any(parameter.requires_grad for parameter in backbone_params)


def test_unfreeze_last_stages_and_all_parameters() -> None:
    model = build_transfer_model(
        "resnet50",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )

    unfreeze_last_stages(model, num_stages=1)
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())

    freeze_backbone_parameters(model)
    assert not any(parameter.requires_grad for parameter in model.layer4.parameters())

    unfreeze_all_parameters(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_unfreeze_last_stages_is_partial_for_feature_backbones() -> None:
    model = build_transfer_model(
        "efficientnet_b0",
        num_classes=3,
        pretrained=False,
        freeze_backbone=True,
    )

    unfreeze_last_stages(model, num_stages=1)

    assert any(parameter.requires_grad for parameter in model.features[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.features[0].parameters())


def test_build_model_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        build_model("baseline_cnn", num_classes=0)
