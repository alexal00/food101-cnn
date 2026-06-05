import csv
import json
from pathlib import Path

import torch
from torch import nn

from food101_cnn.evaluation.calibration import (
    calibration_report,
    save_calibration_report,
)
from food101_cnn.evaluation.confusion import (
    compute_confusion_matrix,
    save_confusion_matrix_csv,
)
from food101_cnn.evaluation.errors import (
    collect_misclassifications,
    copy_misclassified_images,
    save_misclassifications_csv,
)
from food101_cnn.evaluation.latency import measure_inference_latency


def test_confusion_matrix_and_csv_output(tmp_path: Path) -> None:
    predictions = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])

    matrix = compute_confusion_matrix(predictions, targets, num_classes=3)
    output_csv = save_confusion_matrix_csv(
        matrix,
        ["apple_pie", "pizza", "ramen"],
        tmp_path / "confusion.csv",
    )

    assert matrix.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]
    assert output_csv.is_file()
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("true_label")


def test_calibration_report_and_json_output(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0],
            [2.0, 1.0],
            [0.5, 1.5],
        ]
    )
    targets = torch.tensor([0, 1, 1])

    report = calibration_report(logits, targets, num_bins=4)
    output_json = save_calibration_report(report, tmp_path / "calibration.json")
    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert 0 <= report.ece <= 1
    assert report.total == 3
    assert len(report.bins) == 4
    assert payload["num_bins"] == 4


def test_misclassification_csv_and_image_copy(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image content")
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ]
    )
    targets = torch.tensor([0, 2])

    records = collect_misclassifications(
        logits,
        targets,
        class_names=["apple_pie", "pizza", "ramen"],
        image_paths=[image_path, image_path],
        top_k=2,
    )
    output_csv = save_misclassifications_csv(records, tmp_path / "errors.csv")
    copied = copy_misclassified_images(records, tmp_path / "copied")

    assert len(records) == 1
    assert records[0].true_label == "ramen"
    assert records[0].predicted_label == "pizza"
    assert copied and copied[0].is_file()

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["predicted_label"] == "pizza"
    assert json.loads(rows[0]["top_k_labels"]) == ["pizza", "ramen"]


def test_latency_measurement_returns_positive_values() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    sample_batch = torch.randn(2, 3, 4, 4)

    result = measure_inference_latency(
        model,
        sample_batch,
        "cpu",
        warmup=1,
        repeats=2,
    )

    assert result.mean_ms > 0
    assert result.repeats == 2
    assert result.batch_size == 2
    assert result.throughput_items_per_second > 0
