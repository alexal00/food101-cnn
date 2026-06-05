"""High-level training orchestration utilities."""

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, Optimizer

from food101_cnn.training.callbacks import (
    CheckpointManager,
    EarlyStopping,
    create_summary_writer,
    log_epoch_metrics,
)
from food101_cnn.training.loops import EpochMetrics, train_one_epoch, validate_one_epoch
from food101_cnn.training.losses import build_classification_loss
from food101_cnn.training.schedulers import build_scheduler

LOGGER = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
) -> Optimizer:
    """Build the configured optimizer."""
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")

    normalized = optimizer_name.lower()
    if normalized == "adam":
        return Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def fit(
    *,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: dict[str, Any],
    device: torch.device | str,
    epochs: int,
    lr: float,
    checkpoint_dir: str | Path,
    checkpoint_filename: str = "best_model.pt",
    tensorboard_dir: str | Path | None = None,
    max_batches: int | None = None,
    log_progress: bool = True,
    resume_checkpoint: str | Path | None = None,
) -> list[dict[str, EpochMetrics]]:
    """Run a compact training loop for one stage."""
    training_config = config["training"]
    criterion = build_classification_loss(
        label_smoothing=float(training_config.get("label_smoothing", 0.0))
    )
    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        optimizer_name=str(training_config.get("optimizer", "adam")),
    )
    scheduler = build_scheduler(
        optimizer,
        str(training_config.get("scheduler", "reduce_on_plateau")),
        mode="max",
        patience=max(1, int(training_config.get("early_stopping_patience", 6)) // 2),
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        metric_name="val_top5",
        mode="max",
        filename=checkpoint_filename,
    )
    early_stopping = EarlyStopping(
        patience=int(training_config.get("early_stopping_patience", 6)),
        mode="max",
    )
    writer = create_summary_writer(tensorboard_dir) if tensorboard_dir else None
    history: list[dict[str, EpochMetrics]] = []
    start_epoch = 0

    if resume_checkpoint is not None:
        checkpoint = torch.load(Path(resume_checkpoint).expanduser().resolve(), map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Resume checkpoint must be a state_dict or contain model_state_dict.")
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                _move_optimizer_state_to_device(optimizer, torch.device(device))
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("metric_name") == checkpoint_manager.metric_name:
                checkpoint_manager.best_score = float(checkpoint["metric_value"])
                early_stopping.best_score = float(checkpoint["metric_value"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1

    if log_progress:
        LOGGER.info(
            "training_start epochs=%s start_epoch=%s device=%s lr=%.6g optimizer=%s "
            "scheduler=%s mixed_precision=%s max_batches=%s resume_checkpoint=%s",
            epochs,
            start_epoch,
            device,
            lr,
            training_config.get("optimizer", "adam"),
            training_config.get("scheduler", "reduce_on_plateau"),
            bool(training_config.get("mixed_precision", False)),
            max_batches,
            resume_checkpoint,
        )

    try:
        for epoch_offset in range(epochs):
            epoch = start_epoch + epoch_offset
            epoch_start = time.perf_counter()
            if log_progress:
                LOGGER.info(
                    "epoch=%s/%s status=started",
                    epoch_offset + 1,
                    epochs,
                )

            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                mixed_precision=bool(training_config.get("mixed_precision", False)),
                gradient_accumulation_steps=int(
                    training_config.get("gradient_accumulation_steps", 1)
                ),
                max_batches=max_batches,
            )
            val_metrics = validate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                max_batches=max_batches,
            )
            if scheduler is not None:
                scheduler.step(val_metrics.top5)

            checkpoint_saved = checkpoint_manager.step(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_value=val_metrics.top5,
                scheduler=scheduler,
                config=config,
            )

            learning_rate = optimizer.param_groups[0]["lr"]
            if writer is not None:
                log_epoch_metrics(
                    writer,
                    train_metrics,
                    prefix="train",
                    epoch=epoch,
                    learning_rate=learning_rate,
                )
                log_epoch_metrics(writer, val_metrics, prefix="validation", epoch=epoch)

            history.append({"train": train_metrics, "validation": val_metrics})
            should_stop = early_stopping.step(val_metrics.top5)
            if log_progress:
                duration_seconds = time.perf_counter() - epoch_start
                LOGGER.info(
                    "epoch=%s/%s status=finished train_loss=%.6f train_top1=%.4f "
                    "train_top5=%.4f val_loss=%.6f val_top1=%.4f val_top5=%.4f "
                    "lr=%.6g checkpoint=%s bad_epochs=%s duration_sec=%.2f",
                    epoch_offset + 1,
                    epochs,
                    train_metrics.loss,
                    train_metrics.top1,
                    train_metrics.top5,
                    val_metrics.loss,
                    val_metrics.top1,
                    val_metrics.top5,
                    learning_rate,
                    "updated" if checkpoint_saved else "unchanged",
                    early_stopping.bad_epochs,
                    duration_seconds,
                )
            if should_stop:
                if log_progress:
                    LOGGER.info(
                        "early_stopping_triggered epoch=%s best_val_top5=%.4f",
                        epoch_offset + 1,
                        early_stopping.best_score or 0.0,
                    )
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if log_progress:
        LOGGER.info("training_complete epochs_completed=%s", len(history))

    return history


def _move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    """Move loaded optimizer tensors to the active training device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            state[key] = _move_optimizer_value_to_device(value, device)


def _move_optimizer_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_optimizer_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_optimizer_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_value_to_device(item, device) for item in value)
    return value
