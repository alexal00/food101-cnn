"""Training-curve plotting and rule-based diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRIC_SPECS = {
    "loss": ("loss_curves.png", "Training and Validation Loss", "Loss"),
    "top1": ("top1_accuracy_curves.png", "Top-1 Accuracy", "Accuracy"),
    "top5": ("top5_accuracy_curves.png", "Top-5 Accuracy", "Accuracy"),
    "macro_f1": ("macro_f1_curves.png", "Macro F1", "Macro F1"),
    "learning_rate": ("learning_rate_curves.png", "Learning Rate", "Learning rate"),
}
REQUIRED_FIGURE_METRICS = {"loss", "top1", "top5", "macro_f1"}


def save_training_curve_plots(
    scalars: pd.DataFrame,
    output_dir: str | Path = "outputs/figures/training_curves",
) -> dict[str, Path]:
    """Save standard training-curve plots from TensorBoard scalar data."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if scalars.empty:
        return {}

    enriched = _with_metric_columns(scalars)
    generated: dict[str, Path] = {}
    for metric_key, (filename, title, ylabel) in METRIC_SPECS.items():
        metric_frame = enriched[enriched["metric_key"] == metric_key]
        if metric_frame.empty:
            if metric_key in REQUIRED_FIGURE_METRICS:
                generated[metric_key] = _plot_missing_metric(
                    directory / filename,
                    title=title,
                    message=f"No logged scalar tags were found for {title}.",
                )
            continue
        generated[metric_key] = _plot_metric(
            metric_frame,
            directory / filename,
            title=title,
            ylabel=ylabel,
        )

    overview_path = _plot_metric_overview(enriched, directory / "all_scalar_metrics_overview.png")
    if overview_path is not None:
        generated["overview"] = overview_path
    return generated


def analyze_training_diagnostics(scalars: pd.DataFrame) -> dict[str, Any]:
    """Return conservative overfitting/underfitting diagnostics from scalar trends."""
    if scalars.empty:
        return {
            "status": "insufficient_data",
            "evidence": ["No TensorBoard scalar data was available."],
            "likely_causes": ["Training logs were not generated or were not copied locally."],
            "recommended_actions": ["Run training with TensorBoard logging enabled."],
        }

    enriched = _with_metric_columns(scalars)
    train_loss = _mean_curve(enriched, "loss", "train")
    val_loss = _mean_curve(enriched, "loss", "validation")
    train_acc = _mean_curve(enriched, "top5", "train")
    if train_acc.empty:
        train_acc = _mean_curve(enriched, "top1", "train")
    val_acc = _mean_curve(enriched, "top5", "validation")
    if val_acc.empty:
        val_acc = _mean_curve(enriched, "top1", "validation")

    evidence: list[str] = []
    causes: list[str] = []
    actions: list[str] = []
    status = "no_major_issue_detected"

    train_loss_down = _relative_delta(train_loss) < -0.05
    val_loss_delta = _relative_delta(val_loss)
    val_loss_flat_or_up = val_loss_delta > -0.02
    train_acc_up = _absolute_delta(train_acc) > 0.03
    val_acc_delta = _absolute_delta(val_acc)
    val_acc_flat_or_down = val_acc_delta < 0.02

    if train_loss_down:
        evidence.append("Training loss decreases across the logged epochs.")
    if val_loss_flat_or_up:
        evidence.append("Validation loss is flat or increasing across the logged epochs.")
    if train_acc_up and val_acc_flat_or_down:
        evidence.append("Training accuracy improves more clearly than validation accuracy.")

    if train_loss_down and val_loss_flat_or_up and train_acc_up and val_acc_flat_or_down:
        status = "likely_overfitting"
        causes.extend(
            [
                "The classifier may be adapting to training-specific visual patterns.",
                "The current regularization or early-stopping point may be too weak.",
            ]
        )
        actions.extend(
            [
                "increase dropout",
                "increase weight decay",
                "freeze more backbone layers",
                "reduce model capacity",
                "use stronger but still conservative augmentation",
                "stop earlier",
            ]
        )

    low_train_acc = _last_value(train_acc) is not None and float(_last_value(train_acc)) < 0.35
    low_val_acc = _last_value(val_acc) is not None and float(_last_value(val_acc)) < 0.35
    little_loss_progress = abs(_relative_delta(train_loss)) < 0.05 and abs(_relative_delta(val_loss)) < 0.05
    if low_train_acc and low_val_acc and little_loss_progress:
        status = "likely_underfitting"
        evidence.append("Both training and validation accuracy remain low with little loss improvement.")
        causes.extend(
            [
                "The model capacity, learning-rate schedule, or preprocessing may be limiting learning.",
                "Pretrained weights or label/split indexing should be verified.",
            ]
        )
        actions.extend(
            [
                "train longer",
                "increase model capacity",
                "check learning rate",
                "check preprocessing",
                "check labels and split indexing",
                "verify that pretrained weights are loaded",
            ]
        )

    if val_loss_delta > 0.03 and abs(val_acc_delta) < 0.02:
        if status == "no_major_issue_detected":
            status = "possible_overconfidence_or_calibration_issue"
        evidence.append("Validation loss rises while validation accuracy remains roughly stable.")
        causes.append("Predicted probabilities may be overconfident even when top-k ranking is stable.")
        actions.extend(
            [
                "inspect reliability diagram",
                "use or tune label smoothing",
                "apply temperature scaling using validation data",
                "report confidence threshold behavior",
            ]
        )

    if not evidence:
        evidence.append("Logged loss and accuracy curves do not trigger the conservative diagnostic rules.")
    if not causes:
        causes.append("No single dominant failure pattern was detected from scalar trends alone.")
    if not actions:
        actions.append("Inspect class-level errors and calibration before changing training settings.")

    return {
        "status": status,
        "evidence": list(dict.fromkeys(evidence)),
        "likely_causes": list(dict.fromkeys(causes)),
        "recommended_actions": list(dict.fromkeys(actions)),
    }


def save_training_diagnostics(
    diagnostics: dict[str, Any],
    output_dir: str | Path = "outputs/reports",
) -> dict[str, Path]:
    """Save diagnostics to JSON and Markdown."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "training_diagnostics.json"
    md_path = directory / "training_diagnostics.md"
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    md_path.write_text(_diagnostics_to_markdown(diagnostics), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _with_metric_columns(scalars: pd.DataFrame) -> pd.DataFrame:
    required = {"run", "tag", "step", "value"}
    missing = required.difference(scalars.columns)
    if missing:
        raise ValueError(f"Scalar dataframe is missing required columns: {sorted(missing)}")

    enriched = scalars.copy()
    enriched["metric_key"] = enriched["tag"].map(_metric_key)
    enriched["split"] = enriched["tag"].map(_split_name)
    return enriched[enriched["metric_key"].notna()]


def _metric_key(tag: str) -> str | None:
    normalized = _normalize_tag(tag)
    if "loss" in normalized:
        return "loss"
    if any(token in normalized for token in ("top5", "top_5", "top/5")):
        return "top5"
    if any(token in normalized for token in ("top1", "top_1", "top/1")):
        return "top1"
    if "macro" in normalized and "f1" in normalized:
        return "macro_f1"
    if normalized.endswith("/lr") or normalized == "lr" or "learning_rate" in normalized:
        return "learning_rate"
    if "accuracy" in normalized or normalized.endswith("/acc") or "_acc" in normalized:
        return "top1"
    return None


def _split_name(tag: str) -> str:
    normalized = _normalize_tag(tag)
    tokens = set(normalized.replace("_", "/").split("/"))
    if tokens.intersection({"train", "training"}):
        return "train"
    if tokens.intersection({"val", "valid", "validation", "eval"}):
        return "validation"
    return "other"


def _normalize_tag(tag: str) -> str:
    return (
        str(tag)
        .strip()
        .lower()
        .replace("\\", "/")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _plot_metric(metric_frame: pd.DataFrame, output_path: Path, *, title: str, ylabel: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (run, split, tag), group in metric_frame.groupby(["run", "split", "tag"], sort=True):
        ordered = group.sort_values("step")
        label = f"{run} - {split}" if split != "other" else f"{run} - {tag}"
        ax.plot(ordered["step"], ordered["value"], marker="o", linewidth=1.6, markersize=3, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch / step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_missing_metric(output_path: Path, *, title: str, message: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.set_xlabel("Epoch / step")
    ax.set_ylabel("Value")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_metric_overview(enriched: pd.DataFrame, output_path: Path) -> Path | None:
    import matplotlib.pyplot as plt

    metrics = [metric for metric in METRIC_SPECS if metric in set(enriched["metric_key"])]
    if not metrics:
        return None

    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3.8, rows * 3.2)), squeeze=False)
    for axis, metric_key in zip(axes.ravel(), metrics, strict=False):
        frame = enriched[enriched["metric_key"] == metric_key]
        for (run, split), group in frame.groupby(["run", "split"], sort=True):
            ordered = group.sort_values("step")
            axis.plot(ordered["step"], ordered["value"], linewidth=1.2, label=f"{run} - {split}")
        axis.set_title(METRIC_SPECS[metric_key][1])
        axis.set_xlabel("Epoch / step")
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(metrics) :]:
        axis.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
        fig.subplots_adjust(bottom=0.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _mean_curve(enriched: pd.DataFrame, metric_key: str, split: str) -> pd.Series:
    frame = enriched[(enriched["metric_key"] == metric_key) & (enriched["split"] == split)]
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby("step")["value"].mean().sort_index()


def _relative_delta(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    denominator = max(abs(first), 1e-8)
    return (last - first) / denominator


def _absolute_delta(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    return float(series.iloc[-1] - series.iloc[0])


def _last_value(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.iloc[-1])


def _diagnostics_to_markdown(diagnostics: dict[str, Any]) -> str:
    lines = [f"# Training Diagnostics", "", f"Status: `{diagnostics.get('status', 'unknown')}`", ""]
    for title, key in [
        ("Evidence", "evidence"),
        ("Likely causes", "likely_causes"),
        ("Recommended actions", "recommended_actions"),
    ]:
        lines.append(f"## {title}")
        for item in diagnostics.get(key, []):
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)
