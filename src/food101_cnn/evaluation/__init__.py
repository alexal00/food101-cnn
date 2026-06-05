"""Evaluation metrics and analysis utilities."""

from food101_cnn.evaluation.calibration import (
    CalibrationReport,
    calibration_report,
    save_calibration_report,
)
from food101_cnn.evaluation.confusion import (
    compute_confusion_matrix,
    save_confusion_matrix_csv,
)
from food101_cnn.evaluation.errors import (
    MisclassificationRecord,
    collect_misclassifications,
    save_misclassifications_csv,
)
from food101_cnn.evaluation.latency import LatencyResult, measure_inference_latency
from food101_cnn.evaluation.metrics import (
    ClassificationMetrics,
    batch_correct_counts,
    classification_metrics_from_logits,
    macro_f1_score,
    per_class_accuracy,
    top_k_accuracy,
)

__all__ = [
    "CalibrationReport",
    "ClassificationMetrics",
    "LatencyResult",
    "MisclassificationRecord",
    "batch_correct_counts",
    "calibration_report",
    "classification_metrics_from_logits",
    "collect_misclassifications",
    "compute_confusion_matrix",
    "macro_f1_score",
    "measure_inference_latency",
    "per_class_accuracy",
    "save_calibration_report",
    "save_confusion_matrix_csv",
    "save_misclassifications_csv",
    "top_k_accuracy",
]
