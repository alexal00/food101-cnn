"""TensorBoard scalar extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import pandas as pd

SCALAR_COLUMNS = ["run", "tag", "step", "wall_time", "value"]


def load_tensorboard_scalars(log_dir: str | Path) -> pd.DataFrame:
    """Load TensorBoard scalar events into a tidy dataframe.

    The returned schema is ``run, tag, step, wall_time, value``. Missing event
    files return an empty dataframe with that schema. If TensorBoard event
    processing is unavailable, a small CSV-history fallback is attempted.
    """
    base_dir = Path(log_dir).expanduser().resolve()
    if not base_dir.exists():
        return _empty_scalars()

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-not-found]
            EventAccumulator,
        )
    except ImportError as exc:
        fallback = _load_history_csv_fallback(base_dir)
        if fallback is not None:
            return fallback
        raise RuntimeError(
            "TensorBoard is not installed and no training history CSV fallback was "
            "found. Install the project dependencies or provide "
            "outputs/reports/training_history.csv."
        ) from exc

    rows: list[dict[str, object]] = []
    event_files = sorted(base_dir.rglob("events.out.tfevents*"))
    if not event_files:
        fallback = _load_history_csv_fallback(base_dir)
        return fallback if fallback is not None else _empty_scalars()

    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags: Iterable[str] = accumulator.Tags().get("scalars", [])
            run_name = _run_name_from_event_path(event_file, base_dir)
            for tag in scalar_tags:
                for event in accumulator.Scalars(tag):
                    rows.append(
                        {
                            "run": run_name,
                            "tag": tag,
                            "step": int(event.step),
                            "wall_time": float(event.wall_time),
                            "value": float(event.value),
                        }
                    )
        except Exception as exc:  # pragma: no cover - depends on corrupt event files.
            warnings.warn(
                f"Skipping unreadable TensorBoard event file {event_file}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    if not rows:
        return _empty_scalars()
    return pd.DataFrame(rows, columns=SCALAR_COLUMNS).sort_values(
        ["run", "tag", "step", "wall_time"],
        ignore_index=True,
    )


def save_tensorboard_scalars(
    scalars: pd.DataFrame,
    output_csv: str | Path = "outputs/reports/tensorboard_scalars.csv",
) -> Path:
    """Save extracted scalar data to CSV."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_scalar_columns(scalars).to_csv(path, index=False)
    return path


def _empty_scalars() -> pd.DataFrame:
    return pd.DataFrame(columns=SCALAR_COLUMNS)


def _ensure_scalar_columns(scalars: pd.DataFrame) -> pd.DataFrame:
    if scalars.empty:
        return _empty_scalars()
    missing = [column for column in SCALAR_COLUMNS if column not in scalars.columns]
    if missing:
        raise ValueError(f"Scalar dataframe is missing required columns: {missing}")
    return scalars.loc[:, SCALAR_COLUMNS]


def _run_name_from_event_path(event_file: Path, base_dir: Path) -> str:
    event_parent = event_file.parent
    if event_parent.name == "tensorboard" and event_parent.parent != base_dir.parent:
        return event_parent.parent.name

    try:
        relative = event_parent.relative_to(base_dir)
    except ValueError:
        return event_parent.name

    if relative.parts:
        return relative.parts[0]
    return event_parent.name


def _load_history_csv_fallback(base_dir: Path) -> pd.DataFrame | None:
    candidates = [
        base_dir / "training_history.csv",
        base_dir.parent / "reports" / "training_history.csv",
        Path.cwd() / "outputs" / "reports" / "training_history.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _history_csv_to_scalars(candidate)
    return None


def _history_csv_to_scalars(history_csv: Path) -> pd.DataFrame:
    history = pd.read_csv(history_csv)
    if history.empty:
        return _empty_scalars()

    step_column = "step" if "step" in history.columns else "epoch" if "epoch" in history.columns else None
    numeric_columns = [
        column
        for column in history.columns
        if column != step_column and pd.api.types.is_numeric_dtype(history[column])
    ]
    rows: list[dict[str, object]] = []
    for row_index, row in history.iterrows():
        step = int(row[step_column]) if step_column is not None else int(row_index)
        for column in numeric_columns:
            value = row[column]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "run": history_csv.parent.name,
                    "tag": str(column).replace("_", "/"),
                    "step": step,
                    "wall_time": 0.0,
                    "value": float(value),
                }
            )
    if not rows:
        return _empty_scalars()
    return pd.DataFrame(rows, columns=SCALAR_COLUMNS)
