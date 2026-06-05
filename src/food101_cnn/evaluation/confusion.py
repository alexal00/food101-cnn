"""Confusion matrix utilities."""

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

NormalizeMode = Literal["true", None]


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    normalize: NormalizeMode = None,
) -> torch.Tensor:
    """Compute a confusion matrix with rows as true labels and columns as predictions."""
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional.")
    if predictions.size(0) != targets.size(0):
        raise ValueError("predictions and targets batch sizes must match.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for target, prediction in zip(targets.cpu(), predictions.cpu(), strict=True):
        target_index = int(target.item())
        prediction_index = int(prediction.item())
        if 0 <= target_index < num_classes and 0 <= prediction_index < num_classes:
            matrix[target_index, prediction_index] += 1

    if normalize == "true":
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = torch.where(row_sums > 0, matrix / row_sums.clamp_min(1), matrix)
    elif normalize is not None:
        raise ValueError(f"Unsupported normalize mode: {normalize}")

    return matrix


def save_confusion_matrix_csv(
    matrix: torch.Tensor,
    class_names: list[str],
    output_csv: str | Path,
) -> Path:
    """Save a confusion matrix to CSV with class labels."""
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must be square.")
    if matrix.size(0) != len(class_names):
        raise ValueError("class_names length must match matrix dimensions.")

    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *class_names])
        for class_name, row in zip(class_names, matrix.tolist(), strict=True):
            writer.writerow([class_name, *row])

    return path


def save_confusion_matrix_plot(
    matrix: torch.Tensor,
    class_names: list[str],
    output_path: str | Path,
    *,
    max_classes: int | None = 30,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a compact confusion matrix plot for small class subsets."""
    if max_classes is not None and len(class_names) > max_classes:
        raise ValueError(
            f"Refusing to plot {len(class_names)} classes; use CSV for large matrices."
        )
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError(
            "matrix must be square."
        )
    if matrix.size(0) != len(class_names):
        raise ValueError("class_names length must match matrix dimensions.")

    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    size = max(6.0, min(18.0, len(class_names) * 0.32))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.cpu().numpy(), cmap="Blues")
    tick_fontsize = 5 if len(class_names) > 40 else 7 if len(class_names) > 20 else 9
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(range(len(class_names)), labels=class_names, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_confusion_matrix_csv(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a confusion matrix CSV saved by ``save_confusion_matrix_csv``."""
    frame = pd.read_csv(path)
    if frame.empty or frame.columns[0] != "true_label":
        raise ValueError("Confusion CSV must start with a true_label column.")
    class_names = [str(column) for column in frame.columns[1:]]
    matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(class_names):
        raise ValueError("Confusion CSV must contain a square matrix.")
    return matrix, class_names


def normalize_confusion_matrix(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    """Normalize a raw confusion matrix by true-class row support."""
    array = _to_numpy(matrix).astype(float)
    row_sums = array.sum(axis=1, keepdims=True)
    return np.divide(array, row_sums, out=np.zeros_like(array, dtype=float), where=row_sums > 0)


def top_misclassification_pairs(
    matrix: np.ndarray | torch.Tensor,
    class_names: list[str],
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the most common off-diagonal true/predicted class pairs."""
    array = _to_numpy(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")
    if array.shape[0] != len(class_names):
        raise ValueError("class_names length must match matrix dimensions.")

    rows: list[dict[str, object]] = []
    row_sums = array.sum(axis=1)
    for true_index, true_class in enumerate(class_names):
        support = float(row_sums[true_index])
        for pred_index, predicted_class in enumerate(class_names):
            if true_index == pred_index:
                continue
            count = float(array[true_index, pred_index])
            if count <= 0:
                continue
            rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": int(count),
                    "rate_within_true_class": count / support if support else 0.0,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["true_class", "predicted_class", "count", "rate_within_true_class"]
        )
    return pd.DataFrame(rows).sort_values(
        ["count", "rate_within_true_class"],
        ascending=[False, False],
        ignore_index=True,
    ).head(top_n)


def save_confusion_analysis(
    matrix: np.ndarray | torch.Tensor,
    class_names: list[str],
    output_dir: str | Path = "outputs/figures/confusion",
    *,
    best_classes: list[str] | None = None,
    worst_classes: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Path]:
    """Save full/subset confusion plots and top-pair artifacts."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    raw = _to_numpy(matrix)
    normalized = normalize_confusion_matrix(raw)

    outputs: dict[str, Path] = {}
    outputs["full_normalized"] = save_confusion_matrix_plot(
        torch.as_tensor(normalized),
        class_names,
        directory / "confusion_matrix_full_normalized.png",
        max_classes=None,
        title="Full Normalized Confusion Matrix",
    )

    if best_classes:
        outputs["best10_normalized"] = _save_subset_plot(
            normalized,
            class_names,
            best_classes,
            directory / "confusion_matrix_best10_normalized.png",
            title="Best-Performing Classes: Normalized Confusion Matrix",
        )
    if worst_classes:
        outputs["worst10_normalized"] = _save_subset_plot(
            normalized,
            class_names,
            worst_classes,
            directory / "confusion_matrix_worst10_normalized.png",
            title="Worst-Performing Classes: Normalized Confusion Matrix",
        )

    pairs = top_misclassification_pairs(raw, class_names, top_n=top_n)
    pairs_csv = directory.parent.parent / "reports" / "top_misclassification_pairs.csv"
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(pairs_csv, index=False)
    outputs["top_pairs_csv"] = pairs_csv
    outputs["top_pairs_plot"] = save_top_misclassification_pairs_plot(
        pairs,
        directory / "top_misclassification_pairs.png",
    )
    return outputs


def save_top_misclassification_pairs_plot(pairs: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a horizontal bar chart for off-diagonal confusion pairs."""
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * max(len(pairs), 1))))
    if pairs.empty:
        ax.text(0.5, 0.5, "No off-diagonal misclassification pairs", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [
            f"{row.true_class} -> {row.predicted_class}"
            for row in pairs.itertuples(index=False)
        ]
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, pairs["count"].to_numpy(), color="#3f7fbf")
        ax.set_yticks(y_positions, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Most Common Misclassification Pairs")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_subset_plot(
    normalized: np.ndarray,
    class_names: list[str],
    selected_classes: list[str],
    output_path: Path,
    *,
    title: str,
) -> Path:
    indices = [class_names.index(class_name) for class_name in selected_classes if class_name in class_names]
    subset_names = [class_names[index] for index in indices]
    subset = normalized[np.ix_(indices, indices)]
    return save_confusion_matrix_plot(
        torch.as_tensor(subset),
        subset_names,
        output_path,
        max_classes=None,
        title=title,
    )


def _to_numpy(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(matrix):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)
