import pytest
import torch

from food101_cnn.evaluation.metrics import batch_correct_counts, top_k_accuracy
from food101_cnn.evaluation.metrics import (
    classification_metrics_from_logits,
    macro_f1_score,
    per_class_accuracy,
)


def test_top_k_accuracy_computes_top1_and_top3() -> None:
    logits = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )
    targets = torch.tensor([0, 2, 1])

    metrics = top_k_accuracy(logits, targets, top_k=(1, 3))

    assert metrics[1] == pytest.approx(1 / 3)
    assert metrics[3] == pytest.approx(1.0)


def test_batch_correct_counts_caps_k_at_num_classes() -> None:
    logits = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    targets = torch.tensor([1, 0])

    counts = batch_correct_counts(logits, targets, top_k=(1, 5))

    assert counts == {1: 0, 5: 2}


def test_top_k_accuracy_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        top_k_accuracy(torch.empty(0, 3), torch.empty(0, dtype=torch.long))


def test_macro_f1_and_per_class_accuracy() -> None:
    predictions = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])

    macro_f1 = macro_f1_score(predictions, targets, num_classes=3)
    accuracies, support = per_class_accuracy(predictions, targets, num_classes=3)

    assert macro_f1 == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert accuracies == pytest.approx([1.0, 1.0, 0.5])
    assert support == [1, 1, 2]


def test_classification_metrics_from_logits() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 2])

    metrics = classification_metrics_from_logits(
        logits,
        targets,
        num_classes=3,
        top_k=3,
    )

    assert metrics.top1 == pytest.approx(0.75)
    assert metrics.top5 == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
