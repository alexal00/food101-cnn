from pathlib import Path

import food101_cnn


def test_package_importable() -> None:
    assert food101_cnn.__version__ == "0.1.0"


def test_expected_top_level_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/reports",
        "notebooks",
        "outputs/checkpoints",
        "outputs/tensorboard",
        "scripts",
        "src/food101_cnn",
        "tests",
    ]

    missing = [path for path in expected_dirs if not (root / path).is_dir()]
    assert not missing
