"""Reusable training and validation loops."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from food101_cnn.evaluation.metrics import batch_correct_counts


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one epoch."""

    loss: float
    top1: float
    top5: float
    samples: int


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    *,
    top_k: tuple[int, int] = (1, 5),
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> EpochMetrics:
    """Train a model for one epoch and return aggregate metrics."""
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()

    meter = _EpochMeter(top_k=top_k)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = _amp_enabled(torch_device, mixed_precision)
    scaler = _grad_scaler(amp_enabled)
    pending_backward_steps = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs, targets = _move_batch(batch, torch_device)

        with torch.autocast(
            device_type=torch_device.type,
            enabled=amp_enabled,
        ):
            logits = model(inputs)
            loss = criterion(logits, targets)
            scaled_loss = loss / gradient_accumulation_steps

        scaler.scale(scaled_loss).backward()
        pending_backward_steps += 1

        if (batch_index + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending_backward_steps = 0

        meter.update(loss.detach(), logits.detach(), targets.detach())

    if meter.samples == 0:
        raise ValueError("No batches were processed during training.")

    if pending_backward_steps:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return meter.compute()


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: nn.Module,
    device: torch.device | str,
    *,
    top_k: tuple[int, int] = (1, 5),
    max_batches: int | None = None,
) -> EpochMetrics:
    """Evaluate a model for one epoch and return aggregate metrics."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    meter = _EpochMeter(top_k=top_k)

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs, targets = _move_batch(batch, torch_device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        meter.update(loss.detach(), logits.detach(), targets.detach())

    if meter.samples == 0:
        raise ValueError("No batches were processed during validation.")

    return meter.compute()


class _EpochMeter:
    def __init__(self, *, top_k: tuple[int, int]) -> None:
        self.top_k = top_k
        self.loss_sum = 0.0
        self.samples = 0
        self.correct = {k: 0 for k in top_k}

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        batch_size = targets.size(0)
        self.loss_sum += float(loss.item()) * batch_size
        self.samples += batch_size
        counts = batch_correct_counts(logits, targets, top_k=self.top_k)
        for k, count in counts.items():
            self.correct[k] += count

    def compute(self) -> EpochMetrics:
        return EpochMetrics(
            loss=self.loss_sum / self.samples,
            top1=self.correct[self.top_k[0]] / self.samples,
            top5=self.correct[self.top_k[1]] / self.samples,
            samples=self.samples,
        )


def _move_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Expected dataloader batches to contain inputs and targets.")

    inputs, targets = batch[0], batch[1]
    if not torch.is_tensor(inputs) or not torch.is_tensor(targets):
        raise TypeError("Inputs and targets must be torch tensors.")

    non_blocking = device.type == "cuda"
    return (
        inputs.to(device, non_blocking=non_blocking),
        targets.to(device, non_blocking=non_blocking),
    )


def _amp_enabled(device: torch.device, mixed_precision: bool) -> bool:
    return bool(mixed_precision and device.type == "cuda" and torch.cuda.is_available())


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)
