from pathlib import Path

import pandas as pd
from PIL import Image

from food101_cnn.evaluation.qualitative import plot_top5_prediction_panel


def test_plot_top5_prediction_panel(tmp_path: Path) -> None:
    image_paths = []
    for index, color in enumerate([(180, 80, 40), (40, 140, 90)]):
        path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (32, 32), color=color).save(path)
        image_paths.append(path)

    prediction_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "image_path": [str(path) for path in image_paths],
            "true_class": ["apple_pie", "pizza"],
            "pred_class": ["apple_pie", "ramen"],
            "top5_classes": [
                '["apple_pie", "pizza", "ramen"]',
                '["ramen", "pizza", "apple_pie"]',
            ],
            "top5_probabilities": ["[0.8, 0.1, 0.1]", "[0.5, 0.3, 0.2]"],
            "top1_confidence": [0.8, 0.5],
            "is_correct": [True, False],
        }
    ).to_csv(prediction_csv, index=False)

    output = plot_top5_prediction_panel(
        prediction_csv,
        tmp_path,
        ["apple_pie", "pizza", "ramen"],
        tmp_path / "panel.png",
        n_images=2,
        seed=42,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
