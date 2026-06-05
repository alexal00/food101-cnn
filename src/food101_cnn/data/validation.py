"""Image validation utilities for Food-101 files."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

from food101_cnn.data.dataset import Food101Record, load_dataset_index


@dataclass(frozen=True)
class ImageValidationResult:
    """Result from validating one image file."""

    image_path: Path
    is_valid: bool
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    error: str = ""


def validate_image(image_path: str | Path) -> ImageValidationResult:
    """Open, EXIF-transpose, RGB-convert, and load one image."""
    path = Path(image_path).expanduser().resolve()
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image.load()
            width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        return ImageValidationResult(
            image_path=path,
            is_valid=True,
            width=width,
            height=height,
            mode="RGB",
        )
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return ImageValidationResult(image_path=path, is_valid=False, error=str(exc))


def validate_image_paths(
    image_paths: Iterable[str | Path],
    *,
    report_csv: str | Path,
    corrupted_output: str | Path | None = None,
) -> list[ImageValidationResult]:
    """Validate image paths and write a CSV report plus optional failure list."""
    results = [validate_image(path) for path in image_paths]
    write_validation_report(results, report_csv)

    if corrupted_output is not None:
        write_corrupted_images(results, corrupted_output)

    return results


def validate_index(
    index_csv: str | Path,
    *,
    report_csv: str | Path,
    corrupted_output: str | Path | None = None,
) -> list[ImageValidationResult]:
    """Validate all images referenced by a generated dataset index."""
    records = load_dataset_index(index_csv)
    return validate_image_paths(
        (record.image_path for record in records),
        report_csv=report_csv,
        corrupted_output=corrupted_output,
    )


def write_validation_report(
    results: Iterable[ImageValidationResult],
    report_csv: str | Path,
) -> Path:
    """Write image validation results to CSV."""
    path = Path(report_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "is_valid", "width", "height", "mode", "error"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "image_path": str(result.image_path),
                    "is_valid": result.is_valid,
                    "width": result.width if result.width is not None else "",
                    "height": result.height if result.height is not None else "",
                    "mode": result.mode or "",
                    "error": result.error,
                }
            )

    return path


def write_corrupted_images(
    results: Iterable[ImageValidationResult],
    corrupted_output: str | Path,
) -> Path:
    """Write unreadable image paths to a plain-text file."""
    path = Path(corrupted_output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    invalid_paths = [str(result.image_path) for result in results if not result.is_valid]
    path.write_text("\n".join(invalid_paths), encoding="utf-8")
    return path


def validation_summary(results: Iterable[ImageValidationResult]) -> dict[str, int]:
    """Summarize validation results."""
    result_list = list(results)
    invalid = sum(1 for result in result_list if not result.is_valid)
    return {
        "total": len(result_list),
        "valid": len(result_list) - invalid,
        "invalid": invalid,
    }


def validate_records(
    records: Iterable[Food101Record],
    *,
    report_csv: str | Path,
    corrupted_output: str | Path | None = None,
) -> list[ImageValidationResult]:
    """Validate images from in-memory dataset records."""
    return validate_image_paths(
        (record.image_path for record in records),
        report_csv=report_csv,
        corrupted_output=corrupted_output,
    )
