import numpy as np

from food101_cnn.evaluation.class_report import (
    build_classification_report_from_confusion,
    build_classification_report_table,
    top_best_classes,
    top_worst_classes,
)


def test_build_classification_report_table() -> None:
    report = build_classification_report_table(
        y_true=[0, 0, 1, 1, 2],
        y_pred=[0, 1, 1, 2, 2],
        class_names=["apple_pie", "pizza", "ramen"],
    )

    assert list(report["class_name"]) == ["apple_pie", "pizza", "ramen"]
    assert report.loc[0, "support"] == 2
    assert report.loc[0, "correct"] == 1
    assert report.loc[0, "accuracy"] == 0.5
    assert report.loc[2, "accuracy"] == 1.0
    assert top_best_classes(report, top_n=1).iloc[0]["class_name"] == "ramen"
    assert top_worst_classes(report, top_n=1).iloc[0]["class_name"] == "pizza"


def test_build_classification_report_from_confusion() -> None:
    report = build_classification_report_from_confusion(
        np.array([[2, 0], [1, 1]]),
        ["apple_pie", "pizza"],
    )

    assert report.loc[0, "precision"] == 2 / 3
    assert report.loc[1, "recall"] == 0.5
