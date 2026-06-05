"""Classification metrics used during training and validation."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregated classification metrics."""

    top1: float
    top5: float
    macro_f1: float
    per_class_accuracy: list[float]
    per_class_support: list[int]


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: Iterable[int] = (1, 5),
) -> dict[int, float]:
    """Compute top-k accuracies for a batch of logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("logits and targets batch sizes must match.")
    if targets.numel() == 0:
        raise ValueError("Cannot compute accuracy for an empty batch.")

    requested_k = tuple(int(k) for k in top_k)
    if any(k <= 0 for k in requested_k):
        raise ValueError("top_k values must be positive.")

    max_k = min(max(requested_k), logits.size(1))
    predictions = logits.topk(max_k, dim=1).indices
    correct = predictions.eq(targets.view(-1, 1))

    return {
        k: correct[:, : min(k, logits.size(1))].any(dim=1).float().mean().item()
        for k in requested_k
    }


def batch_correct_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: Iterable[int] = (1, 5),
) -> dict[int, int]:
    """Return top-k correct counts for aggregating epoch metrics."""
    accuracies = top_k_accuracy(logits, targets, top_k=top_k)
    batch_size = targets.size(0)
    return {k: int(round(value * batch_size)) for k, value in accuracies.items()}


def macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> float:
    """Compute macro F1 from class predictions and integer targets."""
    _validate_predictions_targets(predictions, targets, num_classes=num_classes)

    f1_values: list[float] = []
    for class_index in range(num_classes):
        pred_positive = predictions == class_index
        target_positive = targets == class_index
        true_positive = torch.logical_and(pred_positive, target_positive).sum().item()
        false_positive = torch.logical_and(pred_positive, ~target_positive).sum().item()
        false_negative = torch.logical_and(~pred_positive, target_positive).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0

        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))

    return float(sum(f1_values) / num_classes)


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[list[float], list[int]]:
    """Compute per-class accuracy and support counts."""
    _validate_predictions_targets(predictions, targets, num_classes=num_classes)

    accuracies: list[float] = []
    support: list[int] = []
    for class_index in range(num_classes):
        class_mask = targets == class_index
        class_support = int(class_mask.sum().item())
        support.append(class_support)
        if class_support == 0:
            accuracies.append(0.0)
        else:
            correct = torch.logical_and(predictions == class_index, class_mask).sum().item()
            accuracies.append(float(correct / class_support))

    return accuracies, support


def classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int | None = None,
    top_k: int = 5,
) -> ClassificationMetrics:
    """Compute top-1, top-k, macro F1, and per-class accuracy."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_classes].")
    resolved_num_classes = num_classes or logits.size(1)
    predictions = logits.argmax(dim=1)
    top_metrics = top_k_accuracy(logits, targets, top_k=(1, top_k))
    class_accuracy, support = per_class_accuracy(
        predictions,
        targets,
        num_classes=resolved_num_classes,
    )
    return ClassificationMetrics(
        top1=top_metrics[1],
        top5=top_metrics[top_k],
        macro_f1=macro_f1_score(
            predictions,
            targets,
            num_classes=resolved_num_classes,
        ),
        per_class_accuracy=class_accuracy,
        per_class_support=support,
    )


def _validate_predictions_targets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
) -> None:
    if predictions.ndim != 1:
        raise ValueError("predictions must have shape [batch_size].")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch_size].")
    if predictions.size(0) != targets.size(0):
        raise ValueError("predictions and targets batch sizes must match.")
    if targets.numel() == 0:
        raise ValueError("Cannot compute metrics for an empty batch.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
