"""Inference latency measurement."""

import math
import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatencyResult:
    """Latency measurements in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_items_per_second: float
    repeats: int
    batch_size: int


def measure_inference_latency(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device | str,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> LatencyResult:
    """Measure forward-pass latency for a fixed sample batch."""
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if sample_batch.ndim < 2:
        raise ValueError("sample_batch must include a batch dimension.")

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    inputs = sample_batch.to(torch_device)

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs)
        _synchronize(torch_device)

        timings: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(inputs)
            _synchronize(torch_device)
            timings.append((time.perf_counter() - start) * 1000)

    sorted_timings = sorted(timings)
    mean_ms = sum(sorted_timings) / repeats
    median_ms = sorted_timings[repeats // 2]
    p95_index = max(0, min(repeats - 1, math.ceil(0.95 * repeats) - 1))
    p95_ms = sorted_timings[p95_index]
    batch_size = int(sample_batch.size(0))
    throughput = batch_size / (mean_ms / 1000)

    return LatencyResult(
        mean_ms=float(mean_ms),
        median_ms=float(median_ms),
        p95_ms=float(p95_ms),
        throughput_items_per_second=float(throughput),
        repeats=repeats,
        batch_size=batch_size,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
