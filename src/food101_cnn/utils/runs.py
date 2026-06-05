"""Run naming, artifact paths, and manifest helpers."""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from food101_cnn import __version__ as package_version

LOCAL_TIMEZONE = "Europe/Paris"
RUN_NAME_PATTERN = re.compile(
    r"^(?P<model>[a-z0-9_]+)_(?P<timestamp>\d{8}-\d{4})_(?P<version>v[0-9A-Za-z_.-]+)$"
)


@dataclass(frozen=True)
class RunPaths:
    """Standard directories and artifact paths for one training run."""

    run_name: str
    run_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    predictions_dir: Path
    tensorboard_dir: Path
    exports_dir: Path
    manifest_path: Path
    best_checkpoint_path: Path


def current_run_timestamp(timezone: str = LOCAL_TIMEZONE) -> str:
    """Return a local timestamp for run names."""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d-%H%M")


def config_model_version(config: dict[str, Any]) -> str:
    """Return the model implementation version configured for a run."""
    version = str(config.get("model", {}).get("version", "")).strip()
    if not version:
        raise ValueError("Config must define model.version, for example v1.")
    return version


def build_run_name(
    config: dict[str, Any],
    *,
    timestamp: str | None = None,
) -> str:
    """Build ``model_YYYYMMDD-HHMM_version`` from config values."""
    model_name = sanitize_run_component(str(config["model"]["name"]))
    version = sanitize_run_component(config_model_version(config))
    run_timestamp = timestamp or current_run_timestamp()
    run_name = f"{model_name}_{run_timestamp}_{version}"
    validate_run_name(run_name)
    return run_name


def validate_run_name(run_name: str) -> None:
    """Validate the agreed run naming convention."""
    if RUN_NAME_PATTERN.match(run_name) is None:
        raise ValueError(
            "Run name must match model_name_YYYYMMDD-HHMM_version, "
            f"got: {run_name}"
        )


def sanitize_run_component(value: str) -> str:
    """Normalize model/version fields for filenames."""
    normalized = value.strip().lower().replace("-", "_")
    normalized = re.sub(r"[^a-z0-9_.]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Run component must not be empty.")
    return normalized


def create_run_paths(
    project_root: str | Path,
    run_name: str,
    *,
    run_root: str | Path = "outputs/runs",
) -> RunPaths:
    """Create and return the standard run directory layout."""
    validate_run_name(run_name)
    root = Path(project_root).resolve()
    root_dir = Path(run_root)
    if not root_dir.is_absolute():
        root_dir = root / root_dir

    run_dir = root_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    predictions_dir = run_dir / "predictions"
    tensorboard_dir = run_dir / "tensorboard"
    exports_dir = run_dir / "exports"
    for directory in (
        checkpoints_dir,
        reports_dir,
        predictions_dir,
        tensorboard_dir,
        exports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_name=run_name,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
        predictions_dir=predictions_dir,
        tensorboard_dir=tensorboard_dir,
        exports_dir=exports_dir,
        manifest_path=run_dir / "manifest.json",
        best_checkpoint_path=checkpoints_dir / f"{run_name}_best_model.pt",
    )


def infer_run_name_from_path(path: str | Path) -> str | None:
    """Infer the run name from a path under ``outputs/runs/<run_name>/``."""
    parts = Path(path).expanduser().resolve().parts
    for index, part in enumerate(parts):
        if part == "runs" and index + 1 < len(parts):
            run_name = parts[index + 1]
            validate_run_name(run_name)
            return run_name
    return None


def parse_run_name(run_name: str) -> dict[str, str]:
    """Parse model, timestamp, and version from a run name."""
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        raise ValueError(f"Invalid run name: {run_name}")
    return match.groupdict()


def write_run_manifest(
    paths: RunPaths,
    *,
    config: dict[str, Any],
    stage: str,
    status: str,
    artifacts: dict[str, str | None] | None = None,
    metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write or update ``manifest.json`` for a run."""
    existing: dict[str, Any] = {}
    if paths.manifest_path.is_file():
        existing = json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    model_config = config.get("model", {})
    payload = {
        **existing,
        "run_name": paths.run_name,
        "model_name": model_config.get("name"),
        "model_version": model_config.get("version"),
        "package_version": package_version,
        "timestamp": parse_run_name(paths.run_name)["timestamp"],
        "timezone": LOCAL_TIMEZONE,
        "stage": stage,
        "status": status,
        "config": config,
        "run_dir": str(paths.run_dir),
        "artifacts": {
            **existing.get("artifacts", {}),
            **(artifacts or {}),
        },
        "metrics": {
            **existing.get("metrics", {}),
            **(metrics or {}),
        },
        "notes": [*existing.get("notes", []), *(notes or [])],
        "updated_at": datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat(timespec="seconds"),
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return paths.manifest_path


def list_run_manifests(
    project_root: str | Path,
    *,
    run_root: str | Path = "outputs/runs",
) -> list[dict[str, Any]]:
    """Load all run manifests sorted by run-name timestamp descending."""
    root = Path(project_root).resolve()
    root_dir = Path(run_root)
    if not root_dir.is_absolute():
        root_dir = root / root_dir
    if not root_dir.is_dir():
        return []

    manifests: list[dict[str, Any]] = []
    for manifest_path in root_dir.glob("*/manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["_manifest_path"] = str(manifest_path)
        manifests.append(payload)

    return sorted(
        manifests,
        key=lambda item: item.get("timestamp", ""),
        reverse=True,
    )


def latest_run_for_model(
    project_root: str | Path,
    model_name: str,
    *,
    run_root: str | Path = "outputs/runs",
) -> dict[str, Any] | None:
    """Return the latest manifest for a model based on filename timestamp."""
    normalized = sanitize_run_component(model_name)
    for manifest in list_run_manifests(project_root, run_root=run_root):
        if sanitize_run_component(str(manifest.get("model_name", ""))) == normalized:
            return manifest
    return None


def select_best_run(
    manifests: list[dict[str, Any]],
    *,
    primary_metric: str = "top5",
    secondary_metric: str = "macro_f1",
) -> dict[str, Any] | None:
    """Select the best run by top-5 accuracy, then macro F1."""
    complete = [
        manifest
        for manifest in manifests
        if manifest.get("status") == "complete" and isinstance(manifest.get("metrics"), dict)
    ]
    if not complete:
        return None

    return max(
        complete,
        key=lambda manifest: (
            _metric_value(manifest, primary_metric),
            _metric_value(manifest, secondary_metric),
            str(manifest.get("timestamp", "")),
        ),
    )


def resolve_manifest_path(
    manifest: dict[str, Any],
    path: str | Path | None,
    *,
    project_root: str | Path | None = None,
    run_root: str | Path = "outputs/runs",
) -> Path | None:
    """Resolve a manifest path, rehoming copied Drive artifacts when needed."""
    if path is None:
        return None

    raw_path = Path(path).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        if project_root is not None:
            candidates.append(Path(project_root).resolve() / raw_path)
        candidates.append(raw_path)

    local_run_dir = _local_manifest_run_dir(
        manifest,
        project_root=project_root,
        run_root=run_root,
    )
    run_name = str(manifest.get("run_name", ""))
    if local_run_dir is not None and run_name:
        path_parts = raw_path.parts
        if run_name in path_parts:
            run_name_index = path_parts.index(run_name)
            relative_tail = Path(*path_parts[run_name_index + 1 :])
            candidates.append(local_run_dir / relative_tail)

    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    return unique_candidates[-1] if unique_candidates else raw_path


def resolve_manifest_artifact(
    manifest: dict[str, Any],
    artifact_name: str,
    *,
    project_root: str | Path | None = None,
    run_root: str | Path = "outputs/runs",
) -> Path | None:
    """Resolve one artifact path from a run manifest."""
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    return resolve_manifest_path(
        manifest,
        artifacts.get(artifact_name),
        project_root=project_root,
        run_root=run_root,
    )


def manifest_with_resolved_artifacts(
    manifest: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    run_root: str | Path = "outputs/runs",
) -> dict[str, Any]:
    """Return a manifest copy with artifact paths resolved for this checkout."""
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return dict(manifest)

    resolved_artifacts = {
        name: str(resolved) if resolved is not None else None
        for name, value in artifacts.items()
        for resolved in [
            resolve_manifest_path(
                manifest,
                value,
                project_root=project_root,
                run_root=run_root,
            )
        ]
    }
    return {**manifest, "artifacts": resolved_artifacts}


def manifest_to_comparison_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """Convert a run manifest to a notebook comparison row."""
    metrics = manifest.get("metrics", {})
    latency = metrics.get("latency", {}) if isinstance(metrics.get("latency"), dict) else {}
    return {
        "model": manifest.get("model_name"),
        "version": manifest.get("model_version"),
        "trained_at": manifest.get("timestamp"),
        "checkpoint": manifest.get("artifacts", {}).get("checkpoint"),
        "top1": metrics.get("top1"),
        "top5": metrics.get("top5"),
        "macro_f1": metrics.get("macro_f1"),
        "ece": metrics.get("ece"),
        "latency_ms": latency.get("mean_ms"),
        "params": metrics.get("params"),
        "notes": "; ".join(manifest.get("notes", [])),
    }


def run_paths_to_dict(paths: RunPaths) -> dict[str, str]:
    """Serialize run paths for tests and manifests."""
    return {key: str(value) for key, value in asdict(paths).items()}


def _metric_value(manifest: dict[str, Any], metric_name: str) -> float:
    value = manifest.get("metrics", {}).get(metric_name)
    if value is None and metric_name == "top5":
        value = manifest.get("metrics", {}).get("val_top5")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _local_manifest_run_dir(
    manifest: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    run_root: str | Path = "outputs/runs",
) -> Path | None:
    manifest_path = manifest.get("_manifest_path")
    if manifest_path:
        return Path(manifest_path).expanduser().resolve().parent

    run_name = manifest.get("run_name")
    if project_root is not None and run_name:
        root = Path(project_root).resolve()
        root_dir = Path(run_root)
        if not root_dir.is_absolute():
            root_dir = root / root_dir
        return root_dir / str(run_name)

    run_dir = manifest.get("run_dir")
    if run_dir:
        return Path(run_dir).expanduser()

    return None
