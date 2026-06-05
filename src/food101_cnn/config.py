"""YAML configuration loading and validation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from food101_cnn.utils.paths import find_project_root, resolve_project_path

ConfigDict = dict[str, Any]

REQUIRED_SECTIONS = (
    "project",
    "data",
    "augmentation",
    "model",
    "training",
    "evaluation",
    "inference",
    "logging",
)

REQUIRED_PATH_FIELDS = {
    "data.root_dir": ("data", "root_dir"),
    "data.processed_dir": ("data", "processed_dir"),
    "logging.tensorboard_dir": ("logging", "tensorboard_dir"),
    "logging.checkpoint_dir": ("logging", "checkpoint_dir"),
    "logging.figures_dir": ("logging", "figures_dir"),
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


def load_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    validate_paths: bool = True,
    create_missing_dirs: bool = False,
) -> ConfigDict:
    """Load and validate a YAML configuration file."""
    path = Path(config_path).expanduser()
    if project_root is not None and not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")

    config = dict(loaded)
    validate_required_sections(config)
    validate_basic_values(config)

    if validate_paths:
        root = Path(project_root).resolve() if project_root else find_project_root(path)
        validate_config_paths(config, project_root=root, create_missing_dirs=create_missing_dirs)

    return config


def validate_required_sections(config: Mapping[str, Any]) -> None:
    """Validate that the expected top-level sections exist."""
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Missing config sections: {', '.join(missing)}")

    non_mappings = [
        section for section in REQUIRED_SECTIONS if not isinstance(config[section], Mapping)
    ]
    if non_mappings:
        raise ConfigError(f"Config sections must be mappings: {', '.join(non_mappings)}")


def validate_basic_values(config: Mapping[str, Any]) -> None:
    """Validate scalar values needed before later pipeline stages run."""
    seed = config["project"].get("seed")
    image_size = config["data"].get("image_size")
    validation_fraction = config["data"].get("validation_fraction")
    num_classes = config["model"].get("num_classes")
    model_version = config["model"].get("version")
    top_k = config["evaluation"].get("top_k")
    threshold = config["inference"].get("threshold")

    if not isinstance(seed, int) or seed < 0:
        raise ConfigError("project.seed must be a non-negative integer.")
    if not isinstance(image_size, int) or image_size <= 0:
        raise ConfigError("data.image_size must be a positive integer.")
    if not isinstance(validation_fraction, (float, int)) or not 0 < validation_fraction < 1:
        raise ConfigError("data.validation_fraction must be between 0 and 1.")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ConfigError("model.num_classes must be a positive integer.")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ConfigError("model.version must be a non-empty string.")
    if not isinstance(top_k, int) or not 1 <= top_k <= num_classes:
        raise ConfigError("evaluation.top_k must be between 1 and model.num_classes.")
    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ConfigError("inference.threshold must be between 0 and 1.")


def validate_config_paths(
    config: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    create_missing_dirs: bool = False,
) -> dict[str, Path]:
    """Resolve and validate configured directories."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    resolved: dict[str, Path] = {}

    for public_name, (section, field) in REQUIRED_PATH_FIELDS.items():
        raw_value = config[section].get(field)
        if not isinstance(raw_value, str) or not raw_value:
            raise ConfigError(f"{public_name} must be a non-empty path string.")

        path = resolve_project_path(raw_value, root)
        if create_missing_dirs:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ConfigError(f"{public_name} directory does not exist: {path}")
        resolved[public_name] = path

    return resolved


def get_config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Return a nested config value using dot notation."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing config key: {dotted_key}")
        current = current[part]
    return current
