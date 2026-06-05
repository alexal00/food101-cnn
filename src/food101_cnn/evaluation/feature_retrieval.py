"""Feature-space nearest-neighbor retrieval utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from food101_cnn.evaluation.errors import resolve_image_path


def extract_hidden_features(
    model: nn.Module,
    dataloader: Any,
    device: torch.device | str,
    layer_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Extract batched hidden features from a named layer or classifier input."""
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    layer = _resolve_feature_layer(model, layer_name)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        captured.append(tensor.detach())

    handle = layer.register_forward_hook(hook)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    try:
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets, batch_paths = _unpack_batch(batch)
                captured.clear()
                _ = model(inputs.to(torch_device))
                if not captured:
                    raise RuntimeError("Feature hook did not capture layer output.")
                batch_features = captured[-1]
                batch_features = torch.flatten(batch_features, start_dim=1)
                features.append(batch_features.cpu().numpy())
                if targets is not None:
                    labels.append(targets.detach().cpu().numpy())
                paths.extend(batch_paths)
    finally:
        handle.remove()

    feature_array = np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)
    label_array = np.concatenate(labels, axis=0) if labels else np.empty((feature_array.shape[0],), dtype=np.int64)
    return {
        "features": feature_array.astype(np.float32, copy=False),
        "labels": label_array,
        "paths": np.asarray(paths, dtype=object),
    }


def find_nearest_feature_neighbors(
    query_features: np.ndarray,
    database_features: np.ndarray,
    top_k: int = 6,
) -> np.ndarray:
    """Return database indices with smallest Euclidean distance per query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    queries = _normalize_rows(np.asarray(query_features, dtype=np.float32))
    database = _normalize_rows(np.asarray(database_features, dtype=np.float32))
    if queries.ndim != 2 or database.ndim != 2:
        raise ValueError("query_features and database_features must be 2D arrays.")
    if queries.shape[1] != database.shape[1]:
        raise ValueError("Feature dimensions must match.")
    if database.shape[0] == 0:
        raise ValueError("database_features must not be empty.")

    distances = np.linalg.norm(queries[:, None, :] - database[None, :, :], axis=2)
    return np.argsort(distances, axis=1)[:, : min(top_k, database.shape[0])]


def save_feature_file(payload: dict[str, np.ndarray], output_npz: str | Path) -> Path:
    """Save extracted feature arrays to an NPZ file."""
    path = Path(output_npz).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def plot_feature_neighbor_panel(
    query_records: pd.DataFrame,
    database_records: pd.DataFrame,
    neighbor_indices: np.ndarray,
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Plot query images beside nearest training-image neighbors."""
    import matplotlib.pyplot as plt

    if len(query_records) != neighbor_indices.shape[0]:
        raise ValueError("query_records length must match neighbor_indices rows.")
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(query_records)
    cols = neighbor_indices.shape[1] + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, max(2.2, rows * 2.1)), squeeze=False)
    for row_index, query in enumerate(query_records.itertuples(index=False)):
        _draw_image_cell(
            axes[row_index, 0],
            getattr(query, "image_path"),
            title=f"query\n{getattr(query, 'class_name', getattr(query, 'true_class', ''))}",
            project_root=project_root,
        )
        query_label = str(getattr(query, "class_name", getattr(query, "true_class", "")))
        for col_offset, database_index in enumerate(neighbor_indices[row_index], start=1):
            neighbor = database_records.iloc[int(database_index)]
            neighbor_label = str(neighbor.get("class_name", neighbor.get("true_class", "")))
            marker = "same" if query_label and neighbor_label == query_label else "diff"
            _draw_image_cell(
                axes[row_index, col_offset],
                str(neighbor["image_path"]),
                title=f"{neighbor_label}\n{marker}",
                project_root=project_root,
            )

    fig.suptitle("Last-Hidden-Layer Feature Nearest Neighbors", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _resolve_feature_layer(model: nn.Module, layer_name: str | None) -> nn.Module:
    if layer_name is not None:
        named_modules = dict(model.named_modules())
        if layer_name not in named_modules:
            raise ValueError(f"Layer not found: {layer_name}")
        return named_modules[layer_name]
    if hasattr(model, "features") and isinstance(model.features, nn.Module):
        return model.features
    modules = [module for module in model.modules() if isinstance(module, nn.Flatten)]
    if modules:
        return modules[-1]
    raise ValueError("Could not infer a hidden feature layer; pass layer_name.")


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
    if not isinstance(batch, (tuple, list)) or not batch:
        raise ValueError("Expected dataloader batch to be a tuple/list.")
    inputs = batch[0]
    if not torch.is_tensor(inputs):
        raise TypeError("Batch inputs must be torch tensors.")
    targets = batch[1] if len(batch) > 1 and torch.is_tensor(batch[1]) else None
    raw_paths = batch[2] if len(batch) > 2 else [""] * inputs.size(0)
    paths = [str(item) for item in raw_paths]
    return inputs, targets, paths


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        return features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def _draw_image_cell(axis, image_path: str, *, title: str, project_root: str | Path | None) -> None:
    resolved = resolve_image_path(image_path, project_root=project_root)
    if resolved is not None:
        axis.imshow(Image.open(resolved).convert("RGB"))
    else:
        axis.imshow(np.ones((224, 224, 3), dtype=float))
        axis.text(0.5, 0.5, "image not local", ha="center", va="center", fontsize=7)
    axis.set_title(title, fontsize=8)
    axis.axis("off")
