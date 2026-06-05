"""Qualitative prediction-panel plotting utilities."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from food101_cnn.evaluation.errors import resolve_image_path


def plot_top5_prediction_panel(
    prediction_csv: str | Path,
    image_root: str | Path | None,
    class_names: Sequence[str],
    output_path: str | Path,
    n_images: int = 8,
    seed: int = 42,
) -> Path:
    """Plot top-5 predictions and probabilities for sampled images."""
    predictions = pd.read_csv(prediction_csv)
    required = {"image_path", "true_class", "top5_classes", "top5_probabilities"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Prediction CSV is empty.")

    sampled = predictions.sample(
        n=min(n_images, len(predictions)),
        random_state=seed,
    ).reset_index(drop=True)
    return plot_top5_prediction_panel_from_frame(
        sampled,
        image_root=image_root,
        class_names=class_names,
        output_path=output_path,
    )


def plot_top5_prediction_panel_from_frame(
    predictions: pd.DataFrame,
    *,
    image_root: str | Path | None,
    class_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Plot a top-5 qualitative panel from an in-memory prediction frame."""
    import matplotlib.pyplot as plt

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    class_name_set = set(str(item) for item in class_names)

    rows = len(predictions)
    fig, axes = plt.subplots(rows, 2, figsize=(9, max(2.4, rows * 2.2)), squeeze=False)
    for row_index, row in enumerate(predictions.itertuples(index=False)):
        image_axis, bar_axis = axes[row_index]
        image_path = _resolve_panel_image(str(row.image_path), image_root=image_root)
        if image_path is not None:
            image_axis.imshow(Image.open(image_path).convert("RGB"))
        else:
            image_axis.imshow(np.ones((224, 224, 3), dtype=float))
            image_axis.text(0.5, 0.5, "image not local", ha="center", va="center", fontsize=8)
        image_axis.set_title(f"correct: {row.true_class}", fontsize=9)
        image_axis.axis("off")

        labels = _parse_sequence(row.top5_classes)
        probabilities = [float(item) for item in _parse_sequence(row.top5_probabilities)]
        labels = [str(label) for label in labels[:5]]
        probabilities = probabilities[: len(labels)]
        true_class = str(row.true_class)
        colors = ["#d62728" if label == true_class else "#4c78a8" for label in labels]
        y_positions = np.arange(len(labels))
        bar_axis.barh(y_positions, probabilities, color=colors)
        bar_axis.set_yticks(y_positions, labels=labels, fontsize=8)
        bar_axis.invert_yaxis()
        bar_axis.set_xlim(0, max(1.0, max(probabilities, default=0.0)))
        bar_axis.set_xlabel("probability", fontsize=8)
        bar_axis.grid(axis="x", alpha=0.25)
        if true_class not in labels:
            bar_axis.text(
                0.02,
                0.96,
                "correct label not in top-5",
                transform=bar_axis.transAxes,
                va="top",
                fontsize=8,
                color="#b00020",
            )
        unknown_labels = [label for label in labels if class_name_set and label not in class_name_set]
        if unknown_labels:
            bar_axis.text(
                0.02,
                0.08,
                "unknown label in CSV",
                transform=bar_axis.transAxes,
                fontsize=7,
                color="#666666",
            )

    fig.suptitle("Top-5 Qualitative Predictions", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _parse_sequence(value: object) -> list:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    if "|" in text and not text.startswith("["):
        return [item for item in text.split("|") if item]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except (SyntaxError, ValueError):
        return [text]


def _resolve_panel_image(raw_path: str, *, image_root: str | Path | None) -> Path | None:
    resolved = resolve_image_path(raw_path, project_root=image_root)
    if resolved is not None:
        return resolved
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path
    if image_root is not None:
        candidate = Path(image_root).expanduser().resolve() / path.name
        if candidate.is_file():
            return candidate
    return None
