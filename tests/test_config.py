import random
from pathlib import Path

import numpy as np
import pytest
import torch

from food101_cnn.config import (
    ConfigError,
    get_config_value,
    load_config,
    validate_config_paths,
)
from food101_cnn.utils.device import select_device
from food101_cnn.utils.seed import seed_everything


def test_load_config_reads_yaml_and_validates_paths() -> None:
    config = load_config("configs/efficientnet_b0.yaml")

    assert config["project"]["seed"] == 42
    assert config["model"]["name"] == "efficientnet_b0"
    assert get_config_value(config, "evaluation.top_k") == 5

    resolved_paths = validate_config_paths(config)
    assert resolved_paths["data.root_dir"].is_dir()
    assert resolved_paths["logging.checkpoint_dir"].is_dir()


def test_load_local_baseline_config() -> None:
    config = load_config("configs/baseline_cnn_local.yaml")

    assert config["model"]["name"] == "baseline_cnn_local"
    assert config["model"]["version"] == "vlocal1"
    assert config["training"]["batch_size"] == 8
    assert config["training"]["device"] == "auto"


def test_load_simple_baseline_config() -> None:
    config = load_config("configs/baseline_cnn_simple.yaml")

    assert config["model"]["name"] == "baseline_cnn_simple"
    assert config["model"]["version"] == "vfull1"
    assert config["training"]["batch_size"] == 64
    assert config["training"]["mixed_precision"] is True
    assert config["training"]["epochs_head"] == 0
    assert config["training"]["epochs_finetune"] == 30


def test_load_config_resolves_relative_path_from_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    config = load_config(
        "configs/baseline_cnn_local.yaml",
        project_root=project_root,
    )

    assert config["model"]["name"] == "baseline_cnn_local"


def test_load_config_rejects_missing_required_section(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("project:\n  seed: 42\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing config sections"):
        load_config(config_path, validate_paths=False)


def test_validate_config_paths_can_create_missing_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: food101_cnn
  seed: 42
data:
  root_dir: data/raw
  processed_dir: data/processed
  use_official_split: true
  validation_fraction: 0.15
  image_size: 224
  cache_resized: true
  correct_exif_orientation: true
augmentation: {}
model:
  name: baseline_cnn
  version: v1
  pretrained: false
  num_classes: 101
  dropout: 0.3
  freeze_backbone_initially: false
training: {}
evaluation:
  top_k: 5
inference:
  threshold: 0.4
logging:
  tensorboard_dir: outputs/tensorboard
  checkpoint_dir: outputs/checkpoints
  figures_dir: outputs/figures
""",
        encoding="utf-8",
    )

    config = load_config(
        config_path,
        project_root=tmp_path,
        create_missing_dirs=True,
    )

    paths = validate_config_paths(config, project_root=tmp_path)
    assert paths["data.processed_dir"] == tmp_path / "data" / "processed"


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_select_device_returns_torch_device() -> None:
    device = select_device("auto")

    assert isinstance(device, torch.device)
    assert device.type in {"mps", "cuda", "cpu"}
