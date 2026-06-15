import csv
from pathlib import Path

from PIL import Image

from food101_cnn.data.dataset import (
    Food101IndexedDataset,
    generate_dataset_index,
    load_dataset_index,
)
from food101_cnn.data.download import (
    has_food101_structure,
    verify_food101_structure,
)
from food101_cnn.data.validation import (
    validate_image_paths,
    validate_index,
    validation_summary,
)


def test_generate_dataset_index_preserves_official_test_split(tmp_path: Path) -> None:
    data_root = _make_tiny_food101(tmp_path)
    index_csv = tmp_path / "dataset_index.csv"

    records = generate_dataset_index(
        data_root,
        output_csv=index_csv,
        validation_fraction=0.5,
        seed=7,
    )

    assert index_csv.is_file()
    assert len(records) == 10
    assert {record.official_split for record in records} == {"train", "test"}
    assert all(record.split == "test" for record in records if record.official_split == "test")

    split_counts = {split: 0 for split in ("train", "val", "test")}
    for record in records:
        split_counts[record.split] += 1

    assert split_counts == {"train": 4, "val": 2, "test": 4}


def test_food101_indexed_dataset_reads_selected_split(tmp_path: Path) -> None:
    data_root = _make_tiny_food101(tmp_path)
    index_csv = tmp_path / "dataset_index.csv"
    generate_dataset_index(data_root, output_csv=index_csv, validation_fraction=0.5)

    dataset = Food101IndexedDataset(index_csv, split="val")
    image, label = dataset[0]

    assert len(dataset) == 2
    assert image.mode == "RGB"
    assert label in {0, 1}


def test_load_dataset_index_relocates_uploaded_cached_paths(tmp_path: Path) -> None:
    cache_dir = tmp_path / "processed" / "f8e106c00724"
    image_path = cache_dir / "images" / "apple_pie" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    _write_image(image_path)
    index_csv = cache_dir / "cached_index.csv"

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "official_split",
                "label",
                "label_index",
                "relative_path",
                "image_path",
                "original_image_path",
                "cached_image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "official_split": "train",
                "label": "apple_pie",
                "label_index": "0",
                "relative_path": "apple_pie/sample.jpg",
                "image_path": "/Users/alex/local/cache/images/apple_pie/sample.jpg",
                "original_image_path": "/Users/alex/local/raw/food-101/images/apple_pie/sample.jpg",
                "cached_image_path": "/Users/alex/local/cache/images/apple_pie/sample.jpg",
            }
        )

    records = load_dataset_index(index_csv)

    assert records[0].image_path == image_path


def test_load_dataset_index_prefers_local_cached_image_over_existing_absolute_path(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "processed" / "f8e106c00724"
    local_image_path = cache_dir / "images" / "apple_pie" / "sample.jpg"
    local_image_path.parent.mkdir(parents=True)
    _write_image(local_image_path)
    drive_image_path = tmp_path / "drive" / "images" / "apple_pie" / "sample.jpg"
    drive_image_path.parent.mkdir(parents=True)
    _write_image(drive_image_path)
    index_csv = cache_dir / "cached_index.csv"

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "official_split",
                "label",
                "label_index",
                "relative_path",
                "image_path",
                "original_image_path",
                "cached_image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "official_split": "train",
                "label": "apple_pie",
                "label_index": "0",
                "relative_path": "apple_pie/sample.jpg",
                "image_path": str(drive_image_path),
                "original_image_path": "/content/drive/raw/food-101/images/apple_pie/sample.jpg",
                "cached_image_path": str(drive_image_path),
            }
        )

    records = load_dataset_index(index_csv)

    assert records[0].image_path == local_image_path


def test_load_dataset_index_relocates_dataset_report_to_repo_raw_data(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "food101-cnn"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    image_path = project_root / "data" / "raw" / "food-101" / "images" / "apple_pie" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    _write_image(image_path)
    index_csv = project_root / "data" / "reports" / "dataset_index.csv"
    index_csv.parent.mkdir(parents=True)

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "official_split",
                "label",
                "label_index",
                "relative_path",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "official_split": "train",
                "label": "apple_pie",
                "label_index": "0",
                "relative_path": "apple_pie/sample.jpg",
                "image_path": "/Users/alex/Documents/IA_aero_upm/food101-cnn/data/raw/food-101/images/apple_pie/sample.jpg",
            }
        )

    records = load_dataset_index(index_csv)

    assert records[0].image_path == image_path


def test_load_dataset_index_uses_committed_processed_cache_when_raw_data_is_missing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "food101-cnn"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    cached_image_path = project_root / "data" / "processed" / "abc123" / "images" / "ramen" / "sample.jpg"
    cached_image_path.parent.mkdir(parents=True)
    _write_image(cached_image_path)
    index_csv = project_root / "data" / "reports" / "dataset_index.csv"
    index_csv.parent.mkdir(parents=True)

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "official_split",
                "label",
                "label_index",
                "relative_path",
                "image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "official_split": "test",
                "label": "ramen",
                "label_index": "0",
                "relative_path": "ramen/sample.jpg",
                "image_path": "/stale/machine/path/food101-cnn/data/raw/food-101/images/ramen/sample.jpg",
            }
        )

    records = load_dataset_index(index_csv)

    assert records[0].image_path == cached_image_path


def test_validate_index_reports_lfs_pointer_cached_image(tmp_path: Path) -> None:
    cache_dir = tmp_path / "processed" / "f8e106c00724"
    pointer_path = cache_dir / "images" / "ramen" / "sample.jpg"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:example\n"
        "size 123\n",
        encoding="utf-8",
    )
    index_csv = cache_dir / "cached_index.csv"

    with index_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "official_split",
                "label",
                "label_index",
                "relative_path",
                "image_path",
                "original_image_path",
                "cached_image_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "official_split": "train",
                "label": "ramen",
                "label_index": "0",
                "relative_path": "ramen/sample.jpg",
                "image_path": str(pointer_path),
                "original_image_path": "/content/drive/raw/food-101/images/ramen/sample.jpg",
                "cached_image_path": str(pointer_path),
            }
        )

    results = validate_index(
        index_csv,
        report_csv=tmp_path / "validation.csv",
        corrupted_output=tmp_path / "corrupted.txt",
    )

    assert validation_summary(results) == {"total": 1, "valid": 0, "invalid": 1}
    assert str(pointer_path.resolve()) in (tmp_path / "corrupted.txt").read_text(
        encoding="utf-8"
    )


def test_verify_food101_structure_reports_counts(tmp_path: Path) -> None:
    data_root = _make_tiny_food101(tmp_path)

    assert has_food101_structure(data_root)
    summary = verify_food101_structure(data_root, expected_classes=2)

    assert summary["class_count"] == 2
    assert summary["train_count"] == 6
    assert summary["test_count"] == 4


def test_validate_images_writes_report_and_corrupted_list(tmp_path: Path) -> None:
    valid_image = tmp_path / "valid.jpg"
    invalid_image = tmp_path / "broken.jpg"
    report_csv = tmp_path / "report.csv"
    corrupted_output = tmp_path / "corrupted.txt"

    Image.new("RGB", (16, 12), color=(100, 20, 30)).save(valid_image)
    invalid_image.write_text("not an image", encoding="utf-8")

    results = validate_image_paths(
        [valid_image, invalid_image],
        report_csv=report_csv,
        corrupted_output=corrupted_output,
    )

    assert validation_summary(results) == {"total": 2, "valid": 1, "invalid": 1}
    assert report_csv.is_file()
    assert str(invalid_image.resolve()) in corrupted_output.read_text(encoding="utf-8")

    with report_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["is_valid"] == "True"
    assert rows[1]["is_valid"] == "False"


def test_validate_index_uses_generated_csv(tmp_path: Path) -> None:
    data_root = _make_tiny_food101(tmp_path)
    index_csv = tmp_path / "dataset_index.csv"
    report_csv = tmp_path / "report.csv"

    generate_dataset_index(data_root, output_csv=index_csv, validation_fraction=0.5)
    results = validate_index(index_csv, report_csv=report_csv)

    assert validation_summary(results) == {"total": 10, "valid": 10, "invalid": 0}


def _make_tiny_food101(tmp_path: Path) -> Path:
    data_root = tmp_path / "raw"
    dataset_dir = data_root / "food-101"
    image_dir = dataset_dir / "images"
    meta_dir = dataset_dir / "meta"
    classes = ["apple_pie", "pizza"]

    for class_name in classes:
        (image_dir / class_name).mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

    train_entries: list[str] = []
    test_entries: list[str] = []
    for class_name in classes:
        for index in range(3):
            stem = f"{class_name}/train_{index}"
            _write_image(image_dir / f"{stem}.jpg")
            train_entries.append(stem)
        for index in range(2):
            stem = f"{class_name}/test_{index}"
            _write_image(image_dir / f"{stem}.jpg")
            test_entries.append(stem)

    (meta_dir / "train.txt").write_text("\n".join(train_entries), encoding="utf-8")
    (meta_dir / "test.txt").write_text("\n".join(test_entries), encoding="utf-8")
    return data_root


def _write_image(path: Path) -> None:
    Image.new("RGB", (20, 12), color=(40, 90, 140)).save(path)
