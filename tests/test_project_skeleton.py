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


def test_obsolete_colab_bundle_transfer_path_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    obsolete_paths = [
        "scripts/create_colab_bundle.py",
        "src/food101_cnn/utils/colab_bundle.py",
        "tests/test_colab_bundle.py",
    ]

    present = [path for path in obsolete_paths if (root / path).exists()]
    assert not present
