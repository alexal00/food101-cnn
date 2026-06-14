from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from food101_cnn.evaluation.feature_retrieval import (
    extract_hidden_features,
    find_nearest_feature_neighbors,
    plot_feature_neighbor_panel,
)


def test_find_nearest_feature_neighbors() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    database = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    neighbors = find_nearest_feature_neighbors(queries, database, top_k=2)

    assert neighbors.tolist() == [[0, 1], [1, 0]]


def test_find_nearest_feature_neighbors_uses_raw_euclidean_distance() -> None:
    queries = np.array([[10.0, 0.0]], dtype=np.float32)
    database = np.array(
        [
            [9.0, 5.0],
            [100.0, 1.0],
        ],
        dtype=np.float32,
    )

    neighbors = find_nearest_feature_neighbors(queries, database, top_k=1)

    assert neighbors.tolist() == [[0]]


def test_extract_hidden_features_from_named_layer() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    dataloader = DataLoader(TensorDataset(torch.ones(2, 1, 2, 2), torch.tensor([0, 1])), batch_size=1)

    payload = extract_hidden_features(model, dataloader, "cpu", layer_name="1")

    assert payload["features"].shape == (2, 3)
    assert payload["labels"].tolist() == [0, 1]


def test_plot_feature_neighbor_panel(tmp_path: Path) -> None:
    image_paths = []
    for index, color in enumerate([(180, 80, 40), (40, 140, 90), (70, 90, 180)]):
        path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (32, 32), color=color).save(path)
        image_paths.append(path)

    query_records = pd.DataFrame(
        {"image_path": [str(image_paths[0])], "class_name": ["apple_pie"]}
    )
    database_records = pd.DataFrame(
        {
            "image_path": [str(image_paths[1]), str(image_paths[2])],
            "class_name": ["apple_pie", "pizza"],
        }
    )
    output = plot_feature_neighbor_panel(
        query_records,
        database_records,
        np.array([[0, 1]]),
        tmp_path / "neighbors.png",
        project_root=tmp_path,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
