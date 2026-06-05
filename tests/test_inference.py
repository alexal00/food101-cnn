import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from food101_cnn.inference.export import (
    export_onnx_model,
    export_pytorch_state,
    load_model_checkpoint,
)
from food101_cnn.inference.predict import (
    iter_input_images,
    load_class_names,
    predict_image,
    predict_paths,
    save_predictions_csv,
)


def test_iter_input_images_supports_files_and_folders(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "nested" / "b.png"
    ignored = tmp_path / "notes.txt"
    image_b.parent.mkdir(parents=True)
    _write_image(image_a)
    _write_image(image_b)
    ignored.write_text("not an image", encoding="utf-8")

    assert iter_input_images(image_a) == [image_a.resolve()]
    assert iter_input_images(tmp_path) == [image_a.resolve(), image_b.resolve()]


def test_predict_image_applies_confidence_threshold(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    model = FixedLogitModel(torch.tensor([[2.0, 1.0, 0.0]]))

    accepted = predict_image(
        model,
        image_path,
        transform=_dummy_transform,
        class_names=["apple_pie", "pizza", "ramen"],
        device="cpu",
        top_k=2,
        threshold=0.5,
    )
    uncertain = predict_image(
        model,
        image_path,
        transform=_dummy_transform,
        class_names=["apple_pie", "pizza", "ramen"],
        device="cpu",
        top_k=2,
        threshold=0.9,
    )

    assert accepted.decision == "accepted"
    assert accepted.predicted_label == "apple_pie"
    assert accepted.top_k_labels == ["apple_pie", "pizza"]
    assert uncertain.decision == "uncertain"
    assert uncertain.predicted_label == "uncertain"


def test_predict_paths_and_csv_output(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    output_csv = tmp_path / "predictions.csv"
    _write_image(image_path)
    model = FixedLogitModel(torch.tensor([[0.0, 3.0, 1.0]]))

    predictions = predict_paths(
        model,
        [image_path],
        transform=_dummy_transform,
        class_names=["apple_pie", "pizza", "ramen"],
        device="cpu",
        top_k=3,
        threshold=0.4,
    )
    save_predictions_csv(predictions, output_csv)

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["decision"] == "accepted"
    assert rows[0]["predicted_label"] == "pizza"
    assert rows[0]["top_k_labels"].split("|")[0] == "pizza"


def test_load_class_names_from_file(tmp_path: Path) -> None:
    class_file = tmp_path / "classes.txt"
    class_file.write_text("apple_pie\npizza\n", encoding="utf-8")

    assert load_class_names(explicit_path=class_file) == ["apple_pie", "pizza"]


def test_export_pytorch_state_and_reload(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    output_path = tmp_path / "model.pt"

    export_pytorch_state(
        model,
        output_path,
        config={"model": {"name": "linear"}},
        class_names=["a", "b"],
    )
    loaded = torch.load(output_path, map_location="cpu")
    restored = nn.Linear(4, 2)
    load_model_checkpoint(restored, output_path)

    assert output_path.is_file()
    assert loaded["class_names"] == ["a", "b"]
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.allclose(left, right)


def test_export_onnx_model_invokes_torch_export(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "model.onnx"
    calls = {}

    def fake_export(model, dummy_input, output, **kwargs):
        calls["shape"] = tuple(dummy_input.shape)
        calls["output_names"] = kwargs["output_names"]
        Path(output).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    exported = export_onnx_model(
        nn.Linear(4, 2),
        output_path,
        input_shape=(1, 4),
        device="cpu",
    )

    assert exported == output_path.resolve()
    assert output_path.read_bytes() == b"onnx"
    assert calls == {"shape": (1, 4), "output_names": ["logits"]}


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.repeat(inputs.size(0), 1)


def _dummy_transform(image: Image.Image) -> torch.Tensor:
    return torch.zeros(3, 4, 4)


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(120, 60, 30)).save(path)
