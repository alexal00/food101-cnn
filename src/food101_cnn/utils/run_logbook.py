"""Run logbook generation from completed run manifests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from food101_cnn.utils.runs import (
    list_run_manifests,
    resolve_manifest_artifact,
    sanitize_run_component,
)


@dataclass(frozen=True)
class RunLogbookEntry:
    """Small committed summary for one completed run."""

    model: str
    run_name: str
    timestamp: str
    config: str
    stage: str
    top1: float | None
    top5: float | None
    macro_f1: float | None
    ece: float | None
    checkpoint: str
    notes: str


def build_run_logbook_entries(
    project_root: str | Path,
    *,
    run_root: str | Path = "outputs/runs",
    latest_only: bool = True,
) -> list[RunLogbookEntry]:
    """Build logbook entries from completed run manifests."""
    root = Path(project_root).resolve()
    manifests = [
        manifest
        for manifest in list_run_manifests(root, run_root=run_root)
        if manifest.get("status") == "complete"
    ]
    if latest_only:
        manifests = _latest_manifest_per_model(manifests)

    entries = [
        manifest_to_logbook_entry(manifest, project_root=root, run_root=run_root)
        for manifest in manifests
    ]
    return sorted(entries, key=lambda entry: (entry.model, entry.timestamp))


def manifest_to_logbook_entry(
    manifest: dict[str, Any],
    *,
    project_root: str | Path,
    run_root: str | Path = "outputs/runs",
) -> RunLogbookEntry:
    """Convert one manifest payload into a compact logbook entry."""
    root = Path(project_root).resolve()
    metrics = manifest.get("metrics", {})
    checkpoint = resolve_manifest_artifact(
        manifest,
        "checkpoint",
        project_root=root,
        run_root=run_root,
    )
    return RunLogbookEntry(
        model=str(manifest.get("model_name", "")),
        run_name=str(manifest.get("run_name", "")),
        timestamp=str(manifest.get("timestamp", "")),
        config=_config_label(manifest),
        stage=str(manifest.get("stage", "")),
        top1=_metric(metrics, "top1", "val_top1"),
        top5=_metric(metrics, "top5", "val_top5"),
        macro_f1=_metric(metrics, "macro_f1"),
        ece=_metric(metrics, "ece"),
        checkpoint=_relative_to_root(checkpoint, root) if checkpoint else "",
        notes="; ".join(str(note) for note in manifest.get("notes", [])),
    )


def render_run_logbook_markdown(entries: list[RunLogbookEntry]) -> str:
    """Render logbook entries as Markdown."""
    lines = [
        "# Food-101 Run Logbook",
        "",
        "This logbook is a lightweight committed index of finished runs. It is generated",
        "from `outputs/runs/*/manifest.json` and intentionally excludes checkpoints,",
        "TensorBoard logs, cached images, generated figures, and incomplete run",
        "directories without `manifest.json`.",
        "",
        "Default scope: latest completed run per model.",
        "",
        "| Model | Run | Date | Config | Stage | Top-1 | Top-5 | Macro F1 | ECE | Checkpoint | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    if entries:
        lines.extend(_entry_row(entry) for entry in entries)
    else:
        lines.append("| _none_ | | | | | | | | | | |")
    lines.append("")
    return "\n".join(lines)


def _latest_manifest_per_model(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        model = sanitize_run_component(str(manifest.get("model_name", "")))
        current = latest.get(model)
        if current is None or str(manifest.get("timestamp", "")) > str(current.get("timestamp", "")):
            latest[model] = manifest
    return list(latest.values())


def _config_label(manifest: dict[str, Any]) -> str:
    config = manifest.get("config", {})
    model = config.get("model", {}) if isinstance(config, dict) else {}
    name = str(model.get("name") or manifest.get("model_name") or "")
    version = str(model.get("version") or manifest.get("model_version") or "")
    return " ".join(part for part in [name, version] if part)


def _metric(metrics: Any, *names: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _entry_row(entry: RunLogbookEntry) -> str:
    return (
        f"| {_md(entry.model)} | {_md(entry.run_name)} | {_md(entry.timestamp)} | "
        f"{_md(entry.config)} | {_md(entry.stage)} | {_fmt_float(entry.top1)} | "
        f"{_fmt_float(entry.top5)} | {_fmt_float(entry.macro_f1)} | "
        f"{_fmt_float(entry.ece)} | `{_md(entry.checkpoint)}` | {_md(entry.notes)} |"
    )


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
