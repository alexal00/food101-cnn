import subprocess
import sys
from pathlib import Path

from food101_cnn.utils.experiment_plan import (
    adaptive_batch_size,
    filter_training_plan,
    get_training_plan,
    plan_batch_size,
)


def test_gpu_default_plan_uses_expected_profiles() -> None:
    plan = get_training_plan("gpu-default")

    assert [item.model for item in plan] == ["baseline_cnn_simple", "baseline_cnn_local"]
    assert plan[0].config == "configs/baseline_cnn_simple.yaml"
    assert plan[1].config == "configs/baseline_cnn_local.yaml"
    assert plan[1].mixed_precision is False


def test_gpu_full_plan_uses_gpu_config_files() -> None:
    plan = get_training_plan("gpu-full")
    configs = {item.config for item in plan}

    assert "configs/baseline_cnn_gpu.yaml" in configs
    assert "configs/efficientnet_b0_gpu.yaml" in configs
    assert "configs/resnet50_gpu.yaml" in configs
    assert "configs/convnext_tiny_gpu.yaml" in configs
    assert not any(config.endswith("_colab.yaml") for config in configs)


def test_filter_training_plan_by_model_name() -> None:
    plan = get_training_plan("gpu-full")

    filtered = filter_training_plan(plan, ["resnet50", "convnext-tiny"])

    assert [item.model for item in filtered] == ["resnet50", "convnext_tiny"]


def test_adaptive_batch_size_uses_memory_thresholds() -> None:
    assert adaptive_batch_size("efficientnet_b0", 0) == 32
    assert adaptive_batch_size("efficientnet_b0", 20) == 48
    assert adaptive_batch_size("efficientnet_b0", 35) == 64


def test_explicit_plan_batch_size_wins() -> None:
    item = get_training_plan("gpu-default")[1]

    assert plan_batch_size(item, memory_gb=35) == 8


def test_run_experiment_plan_dry_run_prints_train_commands(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    command = [
        sys.executable,
        "scripts/run_experiment_plan.py",
        "--plan",
        "gpu-default",
        "--train",
        "--dry-run",
        "--gpu-memory-gb",
        "0",
        "--run-root",
        str(run_root),
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "scripts/train.py" in result.stdout
    assert "configs/baseline_cnn_simple.yaml" in result.stdout
    assert "configs/baseline_cnn_local.yaml" in result.stdout
    assert str(run_root) in result.stdout
