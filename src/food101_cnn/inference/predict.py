"""Prediction helpers for single-image and folder inference."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PredictionResult:
    """Top-k prediction output for one image."""

    image_path: str
    decision: str
    predicted_label: str
    confidence: float
    top_k_labels: list[str]
    top_k_probabilities: list[float]
    threshold: float


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    *,
    transform,
    class_names: list[str],
    device: torch.device | str,
    top_k: int = 5,
    threshold: float = 0.4,
) -> PredictionResult:
    """Predict top-k probabilities for a single image."""
    return predict_paths(
        model,
        [image_path],
        transform=transform,
        class_names=class_names,
        device=device,
        top_k=top_k,
        threshold=threshold,
    )[0]


@torch.no_grad()
def predict_paths(
    model: nn.Module,
    image_paths: Iterable[str | Path],
    *,
    transform,
    class_names: list[str],
    device: torch.device | str,
    top_k: int = 5,
    threshold: float = 0.4,
) -> list[PredictionResult]:
    """Predict top-k probabilities for image paths."""
    if not class_names:
        raise ValueError("class_names must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    resolved_top_k = min(top_k, len(class_names))
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    results: list[PredictionResult] = []
    for image_path in image_paths:
        path = Path(image_path).expanduser().resolve()
        tensor = _load_image_tensor(path, transform).unsqueeze(0).to(torch_device)
        logits = model(tensor)
        probabilities = logits.softmax(dim=1).squeeze(0).cpu()
        top_probabilities, top_indices = probabilities.topk(resolved_top_k)

        labels = [class_names[int(index)] for index in top_indices.tolist()]
        probs = [float(probability) for probability in top_probabilities.tolist()]
        confidence = probs[0]
        is_confident = confidence >= threshold
        results.append(
            PredictionResult(
                image_path=str(path),
                decision="accepted" if is_confident else "uncertain",
                predicted_label=labels[0] if is_confident else "uncertain",
                confidence=confidence,
                top_k_labels=labels,
                top_k_probabilities=probs,
                threshold=threshold,
            )
        )

    return results


def iter_input_images(input_path: str | Path) -> list[Path]:
    """Return sorted image files from a single file or folder."""
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image file extension: {path.suffix}")
        return [path]

    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

    raise FileNotFoundError(f"Input path does not exist: {path}")


def save_predictions_csv(
    predictions: Iterable[PredictionResult],
    output_csv: str | Path,
) -> Path:
    """Save prediction results to CSV."""
    path = Path(output_csv).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "decision",
                "predicted_label",
                "confidence",
                "top_k_labels",
                "top_k_probabilities",
                "threshold",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            row = asdict(prediction)
            row["top_k_labels"] = "|".join(prediction.top_k_labels)
            row["top_k_probabilities"] = "|".join(
                f"{probability:.8f}" for probability in prediction.top_k_probabilities
            )
            writer.writerow(row)

    return path


def load_checkpoint_state(checkpoint_path: str | Path) -> dict:
    """Load a checkpoint and return the model state dict."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint must be a state_dict or contain model_state_dict.")


def load_class_names(
    *,
    explicit_path: str | Path | None = None,
    config: dict | None = None,
    checkpoint_path: str | Path | None = None,
    num_classes: int | None = None,
) -> list[str]:
    """Load class names from an explicit file, checkpoint metadata, or Food-101 metadata."""
    if explicit_path is not None:
        return _read_class_names(Path(explicit_path))

    if checkpoint_path is not None:
        checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("class_names"), list):
            return [str(item) for item in checkpoint["class_names"]]

    if config is not None:
        from food101_cnn.data.dataset import read_food101_classes
        from food101_cnn.data.download import get_food101_dir
        from food101_cnn.utils.paths import find_project_root, resolve_project_path

        data_root = config.get("data", {}).get("root_dir")
        if data_root:
            try:
                root = find_project_root()
                return read_food101_classes(get_food101_dir(resolve_project_path(data_root, root)))
            except (FileNotFoundError, ValueError):
                pass

    if num_classes is None:
        raise ValueError("Could not load class names and num_classes was not provided.")

    return [f"class_{index:03d}" for index in range(num_classes)]


def _load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transform(image)
    if not torch.is_tensor(tensor):
        raise TypeError("transform must return a torch.Tensor.")
    return tensor


def _read_class_names(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    class_names = [
        line.strip()
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"Class names file is empty: {resolved}")
    return class_names
