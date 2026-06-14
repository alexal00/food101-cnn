"""Misclassification analysis utilities."""

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image


@dataclass(frozen=True)
class MisclassificationRecord:
    """Information for one incorrect prediction."""

    image_path: str
    true_label: str
    predicted_label: str
    confidence: float
    top_k_labels: list[str]
    top_k_probabilities: list[float]


@dataclass(frozen=True)
class PredictionRecord:
    """Top-k prediction details for one evaluated image."""

    image_path: str
    true_idx: int
    true_class: str
    pred_idx: int
    pred_class: str
    top1_confidence: float
    top5_classes: list[str]
    top5_probabilities: list[float]
    is_correct: bool


def collect_misclassifications(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_names: list[str],
    image_paths: Iterable[str | Path] | None = None,
    top_k: int = 5,
) -> list[MisclassificationRecord]:
    """Collect top-k information for incorrect top-1 predictions."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("logits and targets batch sizes must match.")
    if logits.size(1) != len(class_names):
        raise ValueError("class_names length must match logits classes.")

    resolved_top_k = min(top_k, logits.size(1))
    probabilities = logits.softmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(resolved_top_k, dim=1)
    predictions = top_indices[:, 0]

    path_list = (
        [str(path) for path in image_paths]
        if image_paths is not None
        else [""] * targets.numel()
    )
    if len(path_list) != targets.numel():
        raise ValueError("image_paths length must match batch size.")

    records: list[MisclassificationRecord] = []
    for index in range(targets.numel()):
        target_index = int(targets[index].item())
        predicted_index = int(predictions[index].item())
        if predicted_index == target_index:
            continue

        labels = [class_names[int(item)] for item in top_indices[index].tolist()]
        probs = [float(item) for item in top_probabilities[index].tolist()]
        records.append(
            MisclassificationRecord(
                image_path=path_list[index],
                true_label=class_names[target_index],
                predicted_label=class_names[predicted_index],
                confidence=probs[0],
                top_k_labels=labels,
                top_k_probabilities=probs,
            )
        )

    return records


def save_misclassifications_csv(
    records: Iterable[MisclassificationRecord],
    output_csv: str | Path,
) -> Path:
    """Save misclassification records to CSV."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "true_label",
                "predicted_label",
                "confidence",
                "top_k_labels",
                "top_k_probabilities",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["top_k_labels"] = json.dumps(record.top_k_labels)
            row["top_k_probabilities"] = json.dumps(record.top_k_probabilities)
            writer.writerow(row)

    return path


def prediction_records_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_names: list[str],
    image_paths: Iterable[str | Path] | None = None,
    top_k: int = 5,
) -> list[PredictionRecord]:
    """Create per-image prediction records from logits and targets."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("logits and targets batch sizes must match.")
    if logits.size(1) != len(class_names):
        raise ValueError("class_names length must match logits classes.")

    resolved_top_k = min(top_k, logits.size(1))
    probabilities = logits.softmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(resolved_top_k, dim=1)
    predictions = top_indices[:, 0]
    path_list = (
        [str(path) for path in image_paths]
        if image_paths is not None
        else [""] * targets.numel()
    )
    if len(path_list) != targets.numel():
        raise ValueError("image_paths length must match batch size.")

    records: list[PredictionRecord] = []
    for index in range(targets.numel()):
        true_idx = int(targets[index].item())
        pred_idx = int(predictions[index].item())
        labels = [class_names[int(item)] for item in top_indices[index].tolist()]
        probs = [float(item) for item in top_probabilities[index].tolist()]
        records.append(
            PredictionRecord(
                image_path=path_list[index],
                true_idx=true_idx,
                true_class=class_names[true_idx],
                pred_idx=pred_idx,
                pred_class=class_names[pred_idx],
                top1_confidence=probs[0],
                top5_classes=labels,
                top5_probabilities=probs,
                is_correct=pred_idx == true_idx,
            )
        )
    return records


def save_prediction_records_csv(
    records: Iterable[PredictionRecord],
    output_csv: str | Path,
) -> Path:
    """Save per-image prediction records to CSV."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "true_idx",
                "true_class",
                "pred_idx",
                "pred_class",
                "top1_confidence",
                "top5_classes",
                "top5_probabilities",
                "is_correct",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["top5_classes"] = json.dumps(record.top5_classes)
            row["top5_probabilities"] = json.dumps(record.top5_probabilities)
            writer.writerow(row)
    return path


def normalize_prediction_dataframe(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction or misclassification CSV columns for post-processing."""
    frame = predictions.copy()
    rename_map = {
        "true_label": "true_class",
        "predicted_label": "pred_class",
        "confidence": "top1_confidence",
        "top_k_labels": "top5_classes",
        "top_k_probabilities": "top5_probabilities",
    }
    frame = frame.rename(columns={key: value for key, value in rename_map.items() if key in frame.columns})
    required = {"image_path", "true_class", "pred_class", "top1_confidence"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction dataframe is missing required columns: {sorted(missing)}")

    if "is_correct" not in frame.columns:
        frame["is_correct"] = frame["true_class"].astype(str) == frame["pred_class"].astype(str)
    else:
        frame["is_correct"] = frame["is_correct"].map(_parse_bool)
    if "top5_classes" not in frame.columns:
        frame["top5_classes"] = frame["pred_class"].map(lambda value: json.dumps([str(value)]))
    if "top5_probabilities" not in frame.columns:
        frame["top5_probabilities"] = frame["top1_confidence"].map(lambda value: json.dumps([float(value)]))

    frame["true_class"] = frame["true_class"].astype(str)
    frame["pred_class"] = frame["pred_class"].astype(str)
    frame["top1_confidence"] = pd.to_numeric(frame["top1_confidence"], errors="coerce").fillna(0.0)
    return frame


def save_misclassified_examples_csv(
    predictions: pd.DataFrame,
    output_csv: str | Path = "outputs/reports/misclassified_examples.csv",
) -> Path:
    """Save normalized misclassified examples to CSV."""
    frame = normalize_prediction_dataframe(predictions)
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[~frame["is_correct"]].to_csv(path, index=False)
    return path


def save_error_galleries(
    predictions: pd.DataFrame,
    output_dir: str | Path = "outputs/figures/errors",
    *,
    class_report: pd.DataFrame | None = None,
    project_root: str | Path | None = None,
    limit: int = 12,
) -> dict[str, Path]:
    """Save high/low-confidence error grids and worst-class examples."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frame = normalize_prediction_dataframe(predictions)
    wrong = frame[~frame["is_correct"]].copy()
    outputs: dict[str, Path] = {}

    outputs["high_confidence_wrong"] = _save_gallery(
        wrong.sort_values("top1_confidence", ascending=False).head(limit),
        directory / "high_confidence_wrong.png",
        "Highest-Confidence Wrong Predictions",
        project_root=project_root,
    )
    outputs["low_confidence_wrong"] = _save_gallery(
        wrong.sort_values("top1_confidence", ascending=True).head(limit),
        directory / "low_confidence_wrong.png",
        "Lowest-Confidence Wrong Predictions",
        project_root=project_root,
    )

    worst_classes = _worst_classes_from_report_or_errors(class_report, wrong)
    worst_examples = wrong[wrong["true_class"].isin(worst_classes)].sort_values(
        ["true_class", "top1_confidence"],
        ascending=[True, False],
    )
    outputs["worst_class_examples"] = _save_gallery(
        worst_examples.head(limit),
        directory / "worst_class_examples.png",
        "Representative Errors from Worst Classes",
        project_root=project_root,
    )

    correct = frame[frame["is_correct"]].sort_values("top1_confidence", ascending=False).head(limit)
    outputs["correct_high_confidence"] = _save_gallery(
        correct,
        directory / "correct_high_confidence.png",
        "High-Confidence Correct Predictions",
        project_root=project_root,
    )
    return outputs


def copy_misclassified_images(
    records: Iterable[MisclassificationRecord],
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    """Copy available misclassified image files to an output directory."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            break

        source = Path(record.image_path)
        if not source.is_file():
            continue

        destination = directory / (
            f"{index:05d}_true-{record.true_label}_pred-{record.predicted_label}"
            f"{source.suffix.lower()}"
        )
        shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def _save_gallery(
    rows: pd.DataFrame,
    output_path: Path,
    title: str,
    *,
    project_root: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4
    count = max(len(rows), 1)
    row_count = int(np.ceil(count / columns))
    fig, axes = plt.subplots(row_count, columns, figsize=(columns * 3.0, row_count * 3.2), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")

    if rows.empty:
        axes.ravel()[0].text(0.5, 0.5, "No examples available", ha="center", va="center")
    else:
        for axis, row in zip(axes.ravel(), rows.itertuples(index=False), strict=False):
            image_path = resolve_image_path(str(row.image_path), project_root=project_root)
            if image_path is not None:
                image = Image.open(image_path).convert("RGB")
                axis.imshow(image)
            else:
                axis.imshow(np.ones((224, 224, 3), dtype=float))
                axis.text(0.5, 0.5, "image not local", ha="center", va="center", fontsize=9)
            axis.set_title(
                f"true: {row.true_class}\npred: {row.pred_class}\nconf: {float(row.top1_confidence):.2f}",
                fontsize=8,
            )
            axis.axis("off")

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def resolve_image_path(raw_image_path: str, *, project_root: str | Path | None = None) -> Path | None:
    """Resolve local Food-101 image paths from local or Colab-style reports."""
    path = Path(raw_image_path).expanduser()
    if path.is_file():
        return path
    if project_root is None:
        return None

    root = Path(project_root).expanduser().resolve()
    candidates: list[Path] = []
    parts = path.parts
    if "images" in parts:
        image_tail = Path(*parts[parts.index("images") + 1 :])
        candidates.append(root / "data" / "raw" / "food-101" / "images" / image_tail)
        for processed_root in sorted((root / "data" / "processed").glob("*/images")):
            candidates.append(processed_root / image_tail)
    if "data" in parts:
        candidates.append(root / Path(*parts[parts.index("data") :]))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _worst_classes_from_report_or_errors(
    class_report: pd.DataFrame | None,
    wrong: pd.DataFrame,
    *,
    top_n: int = 10,
) -> list[str]:
    if class_report is not None and {"class_name", "accuracy"}.issubset(class_report.columns):
        supported = class_report
        if "support" in class_report.columns:
            supported = class_report[class_report["support"] > 0]
        return supported.sort_values("accuracy", ascending=True)["class_name"].astype(str).head(top_n).tolist()
    return wrong["true_class"].value_counts().head(top_n).index.astype(str).tolist()


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}
