import numpy as np

from food101_cnn.evaluation.confusion import (
    normalize_confusion_matrix,
    top_misclassification_pairs,
)


def test_top_misclassification_pairs_excludes_diagonal() -> None:
    pairs = top_misclassification_pairs(
        np.array(
            [
                [8, 2, 1],
                [0, 9, 3],
                [4, 0, 6],
            ]
        ),
        ["apple_pie", "pizza", "ramen"],
        top_n=2,
    )

    assert list(pairs["true_class"]) == ["ramen", "pizza"]
    assert list(pairs["predicted_class"]) == ["apple_pie", "ramen"]
    assert list(pairs["count"]) == [4, 3]
    assert pairs.iloc[0]["rate_within_true_class"] == 0.4


def test_normalize_confusion_matrix_handles_empty_rows() -> None:
    normalized = normalize_confusion_matrix(np.array([[2, 2], [0, 0]]))

    assert normalized.tolist() == [[0.5, 0.5], [0.0, 0.0]]
