"""Training callbacks for checkpointing, early stopping, and logging."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer


Mode = Literal["max", "min"]


@dataclass
class EarlyStopping:
    """Patience-based early stopping state."""

    patience: int
    mode: Mode = "max"
    min_delta: float = 0.0
    best_score: float | None = None
    bad_epochs: int = 0

    def step(self, score: float) -> bool:
        """Update state and return true when training should stop."""
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")

        if self.best_score is None or self._is_improvement(score):
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs > self.patience

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        raise ValueError(f"Unsupported early stopping mode: {self.mode}")


@dataclass
class CheckpointManager:
    """Save best model checkpoints based on a validation metric."""

    checkpoint_dir: Path
    metric_name: str = "top5"
    mode: Mode = "max"
    best_score: float | None = None
    filename: str = "best_model.pt"

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        metric_name: str = "top5",
        mode: Mode = "max",
        filename: str = "best_model.pt",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.metric_name = metric_name
        self.mode = mode
        self.best_score = None
        self.filename = filename

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.filename

    def step(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer | None,
        epoch: int,
        metric_value: float,
        scheduler: Any | None = None,
        config: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> bool:
        """Save a checkpoint if ``metric_value`` improves."""
        if self.best_score is not None and not self._is_improvement(metric_value):
            return False

        self.best_score = metric_value
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric_name=self.metric_name,
            metric_value=metric_value,
            checkpoint_path=self.checkpoint_path,
            scheduler=scheduler,
            config=config,
            extra_state=extra_state,
        )
        return True

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > float(self.best_score)
        if self.mode == "min":
            return score < float(self.best_score)
        raise ValueError(f"Unsupported checkpoint mode: {self.mode}")


def save_checkpoint(
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    epoch: int,
    metric_name: str,
    metric_value: float,
    checkpoint_path: str | Path,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    """Save model and optimizer training state."""
    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "model_state_dict": model.state_dict(),
        "config": config,
        "extra_state": extra_state or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)
    return path


def create_summary_writer(log_dir: str | Path):
    """Create a TensorBoard SummaryWriter."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(Path(log_dir).expanduser().resolve()))


def log_epoch_metrics(
    writer: Any,
    metrics: Any,
    *,
    prefix: str,
    epoch: int,
    learning_rate: float | None = None,
) -> None:
    """Log epoch metrics to TensorBoard."""
    metric_dict = asdict(metrics) if hasattr(metrics, "__dataclass_fields__") else dict(metrics)
    for key, value in metric_dict.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{key}", value, epoch)
    if learning_rate is not None:
        writer.add_scalar("train/learning_rate", learning_rate, epoch)
