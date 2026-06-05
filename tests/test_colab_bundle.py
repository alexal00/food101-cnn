import json
import tarfile
from pathlib import Path

from food101_cnn.utils.colab_bundle import create_colab_bundle


def test_create_colab_bundle_includes_source_and_processed_data(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src" / "food101_cnn").mkdir(parents=True)
    (project_root / "configs").mkdir()
    (project_root / "scripts").mkdir()
    (project_root / "data" / "reports").mkdir(parents=True)
    (project_root / "data" / "processed" / "cache" / "images").mkdir(parents=True)
    (project_root / "data" / "raw").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (project_root / "src" / "food101_cnn" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "configs" / "model.yaml").write_text("project: {}\n", encoding="utf-8")
    (project_root / "scripts" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project_root / "data" / "reports" / "dataset_index.csv").write_text(
        "split,official_split,label,label_index,relative_path,image_path\n",
        encoding="utf-8",
    )
    (project_root / "data" / "processed" / "cache" / "cached_index.csv").write_text(
        "split,official_split,label,label_index,relative_path,image_path\n",
        encoding="utf-8",
    )
    (project_root / "data" / "raw" / "large.txt").write_text("raw", encoding="utf-8")

    output_path = tmp_path / "bundle.tar.gz"
    summary = create_colab_bundle(project_root, output_path)

    assert summary.output_path == output_path.resolve()
    assert summary.total_files >= 5
    assert "data/raw" not in summary.included_roots
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["include_raw_data"] is False

    with tarfile.open(output_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert "food101-cnn/pyproject.toml" in names
    assert "food101-cnn/data/reports/dataset_index.csv" in names
    assert "food101-cnn/data/processed/cache/cached_index.csv" in names
    assert "food101-cnn/data/raw/large.txt" not in names
