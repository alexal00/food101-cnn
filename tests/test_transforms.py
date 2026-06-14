import csv
import json
from pathlib import Path

import torch
from PIL import Image

from food101_cnn.data.cache import cache_resized_images
from food101_cnn.data.dataset import Food101Record, write_dataset_index
from food101_cnn.data.transforms import (
    build_eval_transform,
    build_train_transform,
    resize_with_padding,
)


def test_resize_with_padding_preserves_full_image_context() -> None:
    image = Image.new("RGB", (40, 20), color=(200, 10, 10))

    resized = resize_with_padding(image, size=32)

    assert resized.size == (32, 32)
    assert resized.getpixel((16, 16)) == (200, 10, 10)
    assert resized.getpixel((16, 0)) == (0, 0, 0)


def test_eval_transform_returns_normalized_tensor_shape() -> None:
    transform = build_eval_transform({"data": {"image_size": 64}})
    image = Image.new("RGB", (32, 16), color=(120, 80, 40))

    output = transform(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 64, 64)
    assert output.dtype == torch.float32


def test_train_transform_uses_separate_augmentation_pipeline() -> None:
    config = {
        "data": {"image_size": 48},
        "augmentation": {
            "horizontal_flip_p": 1.0,
            "color_jitter": {
                "brightness": 0.15,
                "contrast": 0.15,
                "saturation": 0.10,
                "hue": 0.03,
            },
            "rotation_degrees": 10,
        },
    }
    transform = build_train_transform(config)

    output = transform(Image.new("RGB", (30, 20), color=(90, 120, 150)))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 48, 48)


def test_cache_resized_images_writes_manifest_and_cached_index(tmp_path: Path) -> None:
    original_image = tmp_path / "apple_pie_0.jpg"
    original_image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 20), color=(20, 100, 200)).save(original_image)

    index_csv = tmp_path / "dataset_index.csv"
    write_dataset_index(
        [
            Food101Record(
                split="train",
                official_split="train",
                label="apple_pie",
                label_index=0,
                relative_path="apple_pie/apple_pie_0.jpg",
                image_path=original_image,
            )
        ],
        index_csv,
    )

    summary = cache_resized_images(index_csv, tmp_path / "processed", image_size=32)

    assert summary.total == 1
    assert summary.cached == 1
    assert summary.skipped == 0
    assert summary.manifest_path.is_file()
    assert summary.cached_index_path.is_file()

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["image_size"] == 32
    assert manifest["correct_exif_orientation"] is True

    with summary.cached_index_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["split"] == "train"
    assert rows[0]["official_split"] == "train"
    assert rows[0]["label"] == "apple_pie"

    cached_image = Image.open(rows[0]["cached_image_path"])
    assert cached_image.size == (32, 32)


def test_cache_resized_images_skips_existing_files(tmp_path: Path) -> None:
    original_image = tmp_path / "pizza_0.jpg"
    Image.new("RGB", (16, 16), color=(20, 100, 200)).save(original_image)

    index_csv = tmp_path / "dataset_index.csv"
    write_dataset_index(
        [
            Food101Record(
                split="val",
                official_split="train",
                label="pizza",
                label_index=1,
                relative_path="pizza/pizza_0.jpg",
                image_path=original_image,
            )
        ],
        index_csv,
    )

    cache_root = tmp_path / "processed"
    first = cache_resized_images(index_csv, cache_root, image_size=32)
    second = cache_resized_images(index_csv, cache_root, image_size=32)

    assert first.cached == 1
    assert second.cached == 0
    assert second.skipped == 1


def test_cache_resized_images_replaces_invalid_existing_files(tmp_path: Path) -> None:
    original_image = tmp_path / "ramen_0.jpg"
    Image.new("RGB", (16, 16), color=(20, 100, 200)).save(original_image)

    index_csv = tmp_path / "dataset_index.csv"
    write_dataset_index(
        [
            Food101Record(
                split="train",
                official_split="train",
                label="ramen",
                label_index=2,
                relative_path="ramen/ramen_0.jpg",
                image_path=original_image,
            )
        ],
        index_csv,
    )

    cache_root = tmp_path / "processed"
    first = cache_resized_images(index_csv, cache_root, image_size=32)
    cached_image = first.cache_dir / "images" / "ramen" / "ramen_0.jpg"
    cached_image.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:example\n"
        "size 123\n",
        encoding="utf-8",
    )

    second = cache_resized_images(index_csv, cache_root, image_size=32)

    assert second.cached == 1
    assert second.skipped == 0
    with Image.open(cached_image) as image:
        assert image.size == (32, 32)
