"""Class-level evaluation report utilities."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_classification_report_table(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Build per-class accuracy, precision, recall, F1, and support metrics."""
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty.")

    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, prediction in zip(true, pred, strict=True):
        if 0 <= target < len(class_names) and 0 <= prediction < len(class_names):
            matrix[target, prediction] += 1
    return build_classification_report_from_confusion(matrix, class_names)


def build_classification_report_from_confusion(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Build class-level metrics from a raw confusion matrix."""
    matrix = np.asarray(confusion_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be square.")
    if matrix.shape[0] != len(class_names):
        raise ValueError("class_names length must match confusion_matrix dimensions.")

    rows: list[dict[str, object]] = []
    for index, class_name in enumerate(class_names):
        true_positive = float(matrix[index, index])
        support = float(matrix[index, :].sum())
        predicted_positive = float(matrix[:, index].sum())
        incorrect = support - true_positive
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_idx": index,
                "class_name": str(class_name),
                "support": int(support),
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct": int(true_positive),
                "incorrect": int(incorrect),
            }
        )
    return pd.DataFrame(rows)


def save_class_report_tables(
    report: pd.DataFrame,
    output_dir: str | Path = "outputs/reports",
    *,
    top_n: int = 10,
) -> dict[str, Path]:
    """Save full, best-class, and worst-class report tables."""
    required = {"class_name", "support", "accuracy", "precision", "recall", "f1", "correct", "incorrect"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"Class report is missing required columns: {sorted(missing)}")

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    full_path = directory / "per_class_metrics.csv"
    best_path = directory / "top10_best_classes.csv"
    worst_path = directory / "top10_worst_classes.csv"

    report.to_csv(full_path, index=False)
    top_best_classes(report, top_n=top_n).to_csv(best_path, index=False)
    top_worst_classes(report, top_n=top_n).to_csv(worst_path, index=False)
    return {"full": full_path, "best": best_path, "worst": worst_path}


def top_best_classes(report: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N classes by true-class accuracy."""
    return report.sort_values(
        ["accuracy", "f1", "support"],
        ascending=[False, False, False],
    ).head(top_n)


def top_worst_classes(report: pd.DataFrame, *, top_n: int = 10) -> pd.DataFrame:
    """Return the bottom-N supported classes by true-class accuracy."""
    supported = report[report["support"] > 0]
    return supported.sort_values(
        ["accuracy", "f1", "support"],
        ascending=[True, True, False],
    ).head(top_n)
