"""Run discovery and model-comparison plotting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from food101_cnn.utils.runs import (
    list_run_manifests,
    manifest_to_comparison_row,
    manifest_with_resolved_artifacts,
)


def discover_available_runs(project_root: str | Path, *, run_root: str | Path = "outputs/runs") -> list[dict[str, Any]]:
    """Load run manifests with artifacts resolved for the local checkout."""
    root = Path(project_root).expanduser().resolve()
    return [
        manifest_with_resolved_artifacts(manifest, project_root=root, run_root=run_root)
        for manifest in list_run_manifests(root, run_root=run_root)
    ]


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    """Load one run manifest JSON file."""
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def load_metrics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load metrics from a manifest or its metrics JSON artifact."""
    metrics = manifest.get("metrics", {})
    if isinstance(metrics, dict) and metrics:
        return metrics
    metrics_path = manifest.get("artifacts", {}).get("metrics_json")
    if metrics_path and Path(metrics_path).is_file():
        return json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {}


def load_predictions(manifest: dict[str, Any]) -> pd.DataFrame | None:
    """Load prediction CSV from a manifest when available."""
    path = manifest.get("artifacts", {}).get("predictions_csv")
    if path and Path(path).is_file():
        return pd.read_csv(path)
    return None


def build_model_comparison_table(manifests: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a top-5 ranked model-comparison table."""
    rows = [manifest_to_comparison_row(manifest) for manifest in manifests]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["top5", "macro_f1"],
        ascending=[False, False],
        na_position="last",
        ignore_index=True,
    )


def latest_manifest_per_model(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the latest manifest per model name."""
    latest: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        model_name = str(manifest.get("model_name", ""))
        if not model_name:
            continue
        current = latest.get(model_name)
        if current is None or str(manifest.get("timestamp", "")) > str(current.get("timestamp", "")):
            latest[model_name] = manifest
    return sorted(latest.values(), key=lambda item: str(item.get("model_name", "")))


def save_global_model_comparison_plots(
    comparison: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save separate top-1, top-5, macro-F1, and latency comparison figures."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if comparison.empty:
        return outputs

    metric_specs = {
        "top1": ("model_top1_accuracy.png", "Top-1 Accuracy", "Accuracy"),
        "top5": ("model_top5_accuracy.png", "Top-5 Accuracy", "Accuracy"),
        "macro_f1": ("model_macro_f1.png", "Macro F1", "Macro F1"),
        "latency_ms": ("model_latency_ms.png", "Inference Latency", "Milliseconds"),
    }
    for metric, (filename, title, ylabel) in metric_specs.items():
        if metric in comparison.columns and comparison[metric].notna().any():
            outputs[metric] = _save_bar_plot(
                comparison.dropna(subset=[metric]),
                metric,
                directory / filename,
                title=title,
                ylabel=ylabel,
            )
    table_path = directory / "model_comparison_summary.csv"
    comparison.to_csv(table_path, index=False)
    outputs["summary_table"] = table_path
    return outputs


def _save_bar_plot(frame: pd.DataFrame, metric: str, output_path: Path, *, title: str, ylabel: str) -> Path:
    import matplotlib.pyplot as plt

    labels = frame["model"].fillna("unknown").astype(str)
    values = frame[metric].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if metric != "latency_ms":
        ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
