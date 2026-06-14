"""Dataset indexing utilities for the official Food-101 split."""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from PIL import Image

from food101_cnn.data.download import get_food101_dir

INDEX_COLUMNS = [
    "split",
    "official_split",
    "label",
    "label_index",
    "relative_path",
    "image_path",
]


@dataclass(frozen=True)
class Food101Record:
    """A single image record in the local Food-101 index."""

    split: str
    official_split: str
    label: str
    label_index: int
    relative_path: str
    image_path: Path


def generate_dataset_index(
    data_root: str | Path,
    *,
    output_csv: str | Path | None = None,
    validation_fraction: float = 0.15,
    seed: int = 42,
    include_missing: bool = False,
) -> list[Food101Record]:
    """Generate a deterministic train/val/test index from official metadata."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    dataset_dir = get_food101_dir(data_root)
    classes = read_food101_classes(dataset_dir)
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    train_records = _read_official_split(
        dataset_dir,
        official_split="train",
        class_to_index=class_to_index,
        split="train",
        include_missing=include_missing,
    )
    test_records = _read_official_split(
        dataset_dir,
        official_split="test",
        class_to_index=class_to_index,
        split="test",
        include_missing=include_missing,
    )
    train_val_records = _assign_validation_split(
        train_records,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    records = sorted(
        [*train_val_records, *test_records],
        key=lambda item: (item.split, item.label_index, item.relative_path),
    )

    if output_csv is not None:
        write_dataset_index(records, output_csv)

    return records


def read_food101_classes(dataset_dir: str | Path) -> list[str]:
    """Read Food-101 class names in stable label-index order."""
    root = Path(dataset_dir).expanduser().resolve()
    classes_path = root / "meta" / "classes.txt"
    if classes_path.is_file():
        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    image_dir = root / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Food-101 image directory not found: {image_dir}")

    return sorted(path.name for path in image_dir.iterdir() if path.is_dir())


def write_dataset_index(records: Iterable[Food101Record], output_csv: str | Path) -> Path:
    """Write dataset records to CSV."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_row(record))

    return path


def load_dataset_index(index_csv: str | Path) -> list[Food101Record]:
    """Load dataset records from a generated CSV index."""
    path = Path(index_csv).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row_to_record(row, index_dir=path.parent) for row in reader]


class Food101IndexedDataset:
    """PyTorch dataset backed by a generated Food-101 CSV index."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        split: str,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.records = [
            record for record in load_dataset_index(index_csv) if record.split == split
        ]
        self.transform = transform

        if not self.records:
            raise ValueError(f"No records found for split: {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[object, int]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert("RGB")
            item = self.transform(image) if self.transform else image.copy()
        return item, record.label_index


def iter_image_paths(records: Iterable[Food101Record]) -> Iterator[Path]:
    """Yield image paths from index records."""
    for record in records:
        yield record.image_path


def record_to_row(record: Food101Record) -> dict[str, str]:
    """Convert a record to a CSV row."""
    return {
        "split": record.split,
        "official_split": record.official_split,
        "label": record.label,
        "label_index": str(record.label_index),
        "relative_path": record.relative_path,
        "image_path": str(record.image_path),
    }


def row_to_record(row: dict[str, str], *, index_dir: str | Path | None = None) -> Food101Record:
    """Convert a CSV row to a record."""
    image_path = _resolve_index_image_path(row, index_dir=index_dir)
    return Food101Record(
        split=row["split"],
        official_split=row["official_split"],
        label=row["label"],
        label_index=int(row["label_index"]),
        relative_path=row["relative_path"],
        image_path=image_path,
    )


def _resolve_index_image_path(
    row: dict[str, str],
    *,
    index_dir: str | Path | None = None,
) -> Path:
    image_path = Path(row["image_path"]).expanduser()
    if index_dir is None:
        return image_path

    base_dir = Path(index_dir).expanduser()
    cached_candidate = base_dir / "images" / row["relative_path"]
    if cached_candidate.is_file():
        return cached_candidate

    if image_path.is_file():
        return image_path

    if not image_path.is_absolute():
        relative_candidate = base_dir / image_path
        if relative_candidate.is_file():
            return relative_candidate

    return image_path


def _read_official_split(
    dataset_dir: Path,
    *,
    official_split: str,
    class_to_index: dict[str, int],
    split: str,
    include_missing: bool,
) -> list[Food101Record]:
    meta_path = dataset_dir / "meta" / f"{official_split}.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Food-101 split metadata not found: {meta_path}")

    records: list[Food101Record] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        relative_without_suffix = line.strip()
        if not relative_without_suffix:
            continue

        label = relative_without_suffix.split("/", maxsplit=1)[0]
        if label not in class_to_index:
            raise ValueError(f"Unknown Food-101 class in metadata: {label}")

        relative_path = f"{relative_without_suffix}.jpg"
        image_path = dataset_dir / "images" / relative_path
        if not include_missing and not image_path.is_file():
            raise FileNotFoundError(f"Indexed image is missing: {image_path}")

        records.append(
            Food101Record(
                split=split,
                official_split=official_split,
                label=label,
                label_index=class_to_index[label],
                relative_path=relative_path,
                image_path=image_path,
            )
        )

    return records


def _assign_validation_split(
    train_records: list[Food101Record],
    *,
    validation_fraction: float,
    seed: int,
) -> list[Food101Record]:
    grouped: dict[str, list[Food101Record]] = defaultdict(list)
    for record in train_records:
        grouped[record.label].append(record)

    rng = random.Random(seed)
    output: list[Food101Record] = []

    for records in grouped.values():
        shuffled = records[:]
        rng.shuffle(shuffled)
        validation_count = _validation_count(len(shuffled), validation_fraction)
        validation_paths = {
            record.relative_path for record in shuffled[:validation_count]
        }

        for record in records:
            split = "val" if record.relative_path in validation_paths else "train"
            output.append(
                Food101Record(
                    split=split,
                    official_split=record.official_split,
                    label=record.label,
                    label_index=record.label_index,
                    relative_path=record.relative_path,
                    image_path=record.image_path,
                )
            )

    return output


def _validation_count(sample_count: int, validation_fraction: float) -> int:
    if sample_count <= 1:
        return 0
    count = max(1, int(sample_count * validation_fraction))
    return min(count, sample_count - 1)
