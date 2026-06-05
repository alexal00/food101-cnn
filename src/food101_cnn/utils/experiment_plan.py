"""Experiment-plan helpers for CLI and notebook orchestration."""

from dataclasses import dataclass
from pathlib import Path

from food101_cnn.utils.runs import sanitize_run_component


@dataclass(frozen=True)
class ExperimentPlanItem:
    """One model stage in a repeatable experiment plan."""

    model: str
    config: str
    epochs: int
    role: str
    batch_size: int | None = None
    mixed_precision: bool = True
    num_workers: int = 2


LOCAL_BASELINE_BATCH_SIZE = 8

GPU_BATCH_SIZE_BY_MEMORY: dict[int, dict[str, int]] = {
    35: {"baseline_cnn": 96, "efficientnet_b0": 64, "resnet50": 48, "convnext_tiny": 32},
    20: {"baseline_cnn": 80, "efficientnet_b0": 48, "resnet50": 40, "convnext_tiny": 28},
    0: {"baseline_cnn": 64, "efficientnet_b0": 32, "resnet50": 32, "convnext_tiny": 24},
}

MODEL_BATCH_PROFILE = {
    "baseline_cnn_local": "baseline_cnn_local",
    "baseline_cnn_simple": "baseline_cnn",
    "baseline_cnn": "baseline_cnn",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
}

GPU_DEFAULT_PLAN = (
    ExperimentPlanItem(
        model="baseline_cnn_simple",
        config="configs/baseline_cnn_simple.yaml",
        epochs=30,
        role="simple CNN architecture with stronger GPU schedule",
    ),
    ExperimentPlanItem(
        model="baseline_cnn_local",
        config="configs/baseline_cnn_local.yaml",
        epochs=10,
        batch_size=LOCAL_BASELINE_BATCH_SIZE,
        mixed_precision=False,
        role="reduced local-style simple CNN baseline",
    ),
)

GPU_FULL_PLAN = (
    *GPU_DEFAULT_PLAN,
    ExperimentPlanItem(
        model="baseline_cnn",
        config="configs/baseline_cnn_gpu.yaml",
        epochs=30,
        role="full custom CNN baseline",
    ),
    ExperimentPlanItem(
        model="efficientnet_b0",
        config="configs/efficientnet_b0_gpu.yaml",
        epochs=40,
        role="main transfer-learning candidate",
    ),
    ExperimentPlanItem(
        model="resnet50",
        config="configs/resnet50_gpu.yaml",
        epochs=35,
        role="classic transfer-learning comparison",
    ),
    ExperimentPlanItem(
        model="convnext_tiny",
        config="configs/convnext_tiny_gpu.yaml",
        epochs=35,
        role="modern CNN transfer-learning comparison",
    ),
)

TRAINING_PLANS = {
    "gpu-default": GPU_DEFAULT_PLAN,
    "gpu-full": GPU_FULL_PLAN,
}


def get_training_plan(name: str) -> tuple[ExperimentPlanItem, ...]:
    """Return a named experiment plan."""
    try:
        return TRAINING_PLANS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TRAINING_PLANS))
        raise KeyError(f"Unknown training plan {name!r}. Supported: {supported}") from exc


def filter_training_plan(
    items: tuple[ExperimentPlanItem, ...],
    model_names: list[str] | None,
) -> tuple[ExperimentPlanItem, ...]:
    """Filter a plan by normalized model names."""
    if not model_names:
        return items
    allowed = {sanitize_run_component(model) for model in model_names}
    return tuple(item for item in items if sanitize_run_component(item.model) in allowed)


def adaptive_batch_size(model_name: str, memory_gb: float) -> int:
    """Select a conservative GPU batch size from detected memory."""
    profile = MODEL_BATCH_PROFILE.get(model_name, model_name)
    if profile == "baseline_cnn_local":
        return LOCAL_BASELINE_BATCH_SIZE

    for threshold in sorted(GPU_BATCH_SIZE_BY_MEMORY, reverse=True):
        if memory_gb >= threshold:
            table = GPU_BATCH_SIZE_BY_MEMORY[threshold]
            if profile in table:
                return table[profile]

    supported = ", ".join(sorted(MODEL_BATCH_PROFILE))
    raise KeyError(f"No batch-size profile for {model_name}. Supported: {supported}")


def plan_batch_size(item: ExperimentPlanItem, memory_gb: float) -> int:
    """Return the explicit or adaptive batch size for a plan item."""
    if item.batch_size is not None:
        return item.batch_size
    return adaptive_batch_size(item.model, memory_gb)


def latest_cached_index(cache_root: str | Path) -> Path | None:
    """Return the most recently modified cached index under a processed-data root."""
    root = Path(cache_root).expanduser()
    candidates = sorted(
        root.glob("*/cached_index.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def selected_index_csv(index_csv: str | Path, cache_root: str | Path) -> Path:
    """Prefer the latest processed-cache index when available."""
    return latest_cached_index(cache_root) or Path(index_csv).expanduser()
