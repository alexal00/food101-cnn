"""Food-101 dataset acquisition helpers."""

from pathlib import Path
from typing import Any

FOOD101_DIR_NAME = "food-101"
EXPECTED_CLASS_COUNT = 101


def get_food101_dir(data_root: str | Path) -> Path:
    """Return the directory containing Food-101 ``images`` and ``meta`` folders."""
    root = Path(data_root).expanduser().resolve()
    if root.name == FOOD101_DIR_NAME:
        return root
    return root / FOOD101_DIR_NAME


def has_food101_structure(data_root: str | Path) -> bool:
    """Return true when the expected Food-101 files are already present."""
    dataset_dir = get_food101_dir(data_root)
    required = [
        dataset_dir / "images",
        dataset_dir / "meta" / "train.txt",
        dataset_dir / "meta" / "test.txt",
    ]
    return all(path.exists() for path in required)


def verify_food101_structure(
    data_root: str | Path,
    *,
    expected_classes: int | None = EXPECTED_CLASS_COUNT,
) -> dict[str, Any]:
    """Verify core Food-101 folders and split metadata."""
    dataset_dir = get_food101_dir(data_root)
    image_dir = dataset_dir / "images"
    meta_dir = dataset_dir / "meta"

    missing = [
        path
        for path in (image_dir, meta_dir / "train.txt", meta_dir / "test.txt")
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Food-101 dataset files: {missing_text}")

    class_dirs = sorted(path for path in image_dir.iterdir() if path.is_dir())
    train_count = _count_nonempty_lines(meta_dir / "train.txt")
    test_count = _count_nonempty_lines(meta_dir / "test.txt")

    if expected_classes is not None and len(class_dirs) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} Food-101 classes, found {len(class_dirs)}."
        )

    return {
        "dataset_dir": dataset_dir,
        "class_count": len(class_dirs),
        "train_count": train_count,
        "test_count": test_count,
    }


def download_food101(
    data_root: str | Path,
    *,
    download: bool = True,
    verify: bool = True,
) -> Path:
    """Download Food-101 with torchvision if it is not already available."""
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if has_food101_structure(root):
        dataset_dir = get_food101_dir(root)
    else:
        if not download:
            raise FileNotFoundError(f"Food-101 dataset not found under {root}")

        from torchvision.datasets import Food101

        Food101(root=str(root), split="train", download=True)
        Food101(root=str(root), split="test", download=True)
        dataset_dir = get_food101_dir(root)

    if verify:
        verify_food101_structure(root)

    return dataset_dir


def _count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
