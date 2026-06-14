"""Disk cache generation for resized Food-101 images."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from food101_cnn.data.dataset import INDEX_COLUMNS, load_dataset_index
from food101_cnn.data.image_ops import DEFAULT_IMAGE_SIZE, resize_with_padding

CACHE_VERSION = "padding_resize_rgb_v1"
CACHED_INDEX_COLUMNS = [
    *INDEX_COLUMNS,
    "original_image_path",
    "cached_image_path",
]


@dataclass(frozen=True)
class CacheSummary:
    """Summary of an image cache generation run."""

    total: int
    cached: int
    skipped: int
    cache_dir: Path
    manifest_path: Path
    cached_index_path: Path


def cache_resized_images(
    index_csv: str | Path,
    cache_root: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    refresh: bool = False,
) -> CacheSummary:
    """Cache EXIF-corrected, RGB, padded-resized images to disk."""
    settings = cache_settings(image_size=image_size)
    cache_dir = Path(cache_root).expanduser().resolve() / preprocessing_hash(settings)
    images_dir = cache_dir / "images"
    manifest_path = cache_dir / "manifest.json"
    cached_index_path = cache_dir / "cached_index.csv"

    records = load_dataset_index(index_csv)
    cached = 0
    skipped = 0
    rows: list[dict[str, Any]] = []

    images_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        output_path = images_dir / record.relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.is_file() and not refresh and _cached_image_is_valid(output_path):
            skipped += 1
        else:
            with Image.open(record.image_path) as image:
                resized = resize_with_padding(image, size=image_size)
                resized.save(output_path, quality=95)
            cached += 1

        rows.append(
            {
                "split": record.split,
                "official_split": record.official_split,
                "label": record.label,
                "label_index": record.label_index,
                "relative_path": record.relative_path,
                "image_path": str(output_path),
                "original_image_path": str(record.image_path),
                "cached_image_path": str(output_path),
            }
        )

    write_cached_index(rows, cached_index_path)
    write_cache_manifest(settings, manifest_path, cached_index_path)

    return CacheSummary(
        total=len(records),
        cached=cached,
        skipped=skipped,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
        cached_index_path=cached_index_path,
    )


def cache_settings(*, image_size: int) -> dict[str, Any]:
    """Return preprocessing settings represented in cache metadata."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    return {
        "cache_version": CACHE_VERSION,
        "image_size": image_size,
        "correct_exif_orientation": True,
        "convert_mode": "RGB",
        "resize": "pad_square",
        "pad_fill": [0, 0, 0],
    }


def preprocessing_hash(settings: dict[str, Any]) -> str:
    """Return a stable short hash for preprocessing settings."""
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def write_cache_manifest(
    settings: dict[str, Any],
    manifest_path: str | Path,
    cached_index_path: str | Path,
) -> Path:
    """Write cache preprocessing metadata."""
    path = Path(manifest_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **settings,
        "preprocessing_hash": preprocessing_hash(settings),
        "cached_index_path": str(Path(cached_index_path).expanduser().resolve()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_cached_index(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    """Write a cached-image index preserving split and label fields."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHED_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _cached_image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False
