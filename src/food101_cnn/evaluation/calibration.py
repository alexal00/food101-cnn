"""Calibration metrics and reports."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration statistics for one confidence bin."""

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    gap: float


@dataclass(frozen=True)
class CalibrationReport:
    """Expected calibration error and bin-level statistics."""

    ece: float
    num_bins: int
    total: int
    bins: list[CalibrationBin]


def calibration_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = 15,
) -> CalibrationReport:
    """Compute expected calibration error from logits and targets."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("logits and targets batch sizes must match.")
    if targets.numel() == 0:
        raise ValueError("Cannot compute calibration for an empty batch.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    probabilities = logits.softmax(dim=1)
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(targets).float()

    bins: list[CalibrationBin] = []
    ece = 0.0
    total = targets.numel()

    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum().item())
        if count == 0:
            accuracy = 0.0
            confidence = 0.0
        else:
            accuracy = float(correctness[mask].mean().item())
            confidence = float(confidences[mask].mean().item())

        gap = abs(accuracy - confidence)
        ece += (count / total) * gap
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                gap=gap,
            )
        )

    return CalibrationReport(
        ece=float(ece),
        num_bins=num_bins,
        total=total,
        bins=bins,
    )


def confidence_calibration_report(
    confidences,
    correctness,
    *,
    num_bins: int = 15,
) -> CalibrationReport:
    """Compute calibration statistics from top-1 confidence and correctness."""
    confidence_array = np.asarray(confidences, dtype=float)
    correctness_array = np.asarray(correctness, dtype=bool)
    if confidence_array.ndim != 1 or correctness_array.ndim != 1:
        raise ValueError("confidences and correctness must be one-dimensional.")
    if confidence_array.shape[0] != correctness_array.shape[0]:
        raise ValueError("confidences and correctness lengths must match.")
    if confidence_array.size == 0:
        raise ValueError("Cannot compute calibration for an empty input.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    bins: list[CalibrationBin] = []
    ece = 0.0
    total = int(confidence_array.size)
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            mask = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            mask = (confidence_array >= lower) & (confidence_array < upper)
        count = int(mask.sum())
        if count:
            accuracy = float(correctness_array[mask].mean())
            confidence = float(confidence_array[mask].mean())
        else:
            accuracy = 0.0
            confidence = 0.0
        gap = abs(accuracy - confidence)
        ece += (count / total) * gap
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                confidence=confidence,
                gap=gap,
            )
        )
    return CalibrationReport(ece=float(ece), num_bins=num_bins, total=total, bins=bins)


def save_calibration_report(report: CalibrationReport, output_json: str | Path) -> Path:
    """Save a calibration report to JSON."""
    path = Path(output_json).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def save_reliability_diagram(
    report: CalibrationReport,
    output_path: str | Path,
) -> Path:
    """Save a reliability diagram from a calibration report."""
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    centers = [(item.lower + item.upper) / 2 for item in report.bins]
    accuracies = [item.accuracy for item in report.bins]
    confidences = [item.confidence for item in report.bins]
    width = 1 / report.num_bins

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.bar(centers, accuracies, width=width * 0.9, alpha=0.7, label="Accuracy")
    ax.plot(centers, confidences, marker="o", label="Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_confidence_histogram(
    confidences,
    output_path: str | Path,
    *,
    num_bins: int = 15,
) -> Path:
    """Save a histogram of top-1 confidence values."""
    import matplotlib.pyplot as plt

    values = np.asarray(confidences, dtype=float)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=np.linspace(0, 1, num_bins + 1), color="#4c78a8", edgecolor="white")
    ax.set_title("Top-1 Confidence Histogram")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Samples")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_calibration_plots_from_predictions(
    predictions: pd.DataFrame,
    figures_dir: str | Path = "outputs/figures/calibration",
    reports_dir: str | Path = "outputs/reports",
    *,
    num_bins: int = 15,
) -> dict[str, Path]:
    """Save reliability, confidence histogram, and ECE JSON from predictions."""
    from food101_cnn.evaluation.errors import normalize_prediction_dataframe

    frame = normalize_prediction_dataframe(predictions)
    report = confidence_calibration_report(
        frame["top1_confidence"].to_numpy(dtype=float),
        frame["is_correct"].to_numpy(dtype=bool),
        num_bins=num_bins,
    )
    fig_dir = Path(figures_dir).expanduser().resolve()
    report_dir = Path(reports_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "reliability": save_reliability_diagram(report, fig_dir / "reliability_diagram.png"),
        "histogram": save_confidence_histogram(
            frame["top1_confidence"].to_numpy(dtype=float),
            fig_dir / "confidence_histogram.png",
            num_bins=num_bins,
        ),
        "metrics": save_calibration_report(report, report_dir / "calibration_metrics.json"),
    }


def load_calibration_report_json(path: str | Path) -> CalibrationReport:
    """Load a calibration report saved by ``save_calibration_report``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bins = [CalibrationBin(**item) for item in payload.get("bins", [])]
    return CalibrationReport(
        ece=float(payload.get("ece", 0.0)),
        num_bins=int(payload.get("num_bins", len(bins) or 1)),
        total=int(payload.get("total", sum(item.count for item in bins))),
        bins=bins,
    )


def save_calibration_artifacts_from_report(
    report: CalibrationReport,
    figures_dir: str | Path = "outputs/figures/calibration",
    reports_dir: str | Path = "outputs/reports",
) -> dict[str, Path]:
    """Save calibration artifacts when only binned report data is available."""
    fig_dir = Path(figures_dir).expanduser().resolve()
    report_dir = Path(reports_dir).expanduser().resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    approximate_confidences = [
        (item.lower + item.upper) / 2
        for item in report.bins
        for _ in range(max(item.count, 0))
    ]
    if not approximate_confidences:
        approximate_confidences = [0.0]
    return {
        "reliability": save_reliability_diagram(report, fig_dir / "reliability_diagram.png"),
        "histogram": save_confidence_histogram(
            approximate_confidences,
            fig_dir / "confidence_histogram.png",
            num_bins=report.num_bins,
        ),
        "metrics": save_calibration_report(report, report_dir / "calibration_metrics.json"),
    }
