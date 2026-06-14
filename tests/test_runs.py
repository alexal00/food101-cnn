import json
import subprocess
import sys
from pathlib import Path

from food101_cnn.config import load_config
from food101_cnn.utils.runs import (
    build_run_name,
    create_run_paths,
    infer_run_name_from_path,
    latest_run_for_model,
    list_incomplete_run_dirs,
    manifest_to_comparison_row,
    manifest_with_resolved_artifacts,
    resolve_manifest_artifact,
    select_best_run,
    write_run_manifest,
)


def test_build_run_name_uses_model_timestamp_and_version() -> None:
    config = load_config("configs/efficientnet_b0.yaml")

    run_name = build_run_name(config, timestamp="20260601-1830")

    assert run_name == "efficientnet_b0_20260601-1830_v1"


def test_create_run_paths_and_manifest(tmp_path: Path) -> None:
    config = load_config("configs/baseline_cnn.yaml")
    run_name = build_run_name(config, timestamp="20260601-1830")
    paths = create_run_paths(tmp_path, run_name)

    manifest_path = write_run_manifest(
        paths,
        config=config,
        stage="training",
        status="complete",
        artifacts={"checkpoint": str(paths.best_checkpoint_path)},
        metrics={"top5": 0.7, "macro_f1": 0.4},
        notes=["synthetic test"],
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_name"] == run_name
    assert payload["model_name"] == "baseline_cnn"
    assert payload["model_version"] == "v1"
    assert payload["artifacts"]["checkpoint"].endswith(f"{run_name}_best_model.pt")
    assert infer_run_name_from_path(paths.best_checkpoint_path) == run_name


def test_infer_run_name_from_colab_run_root_checkpoint() -> None:
    run_name = "baseline_cnn_simple_20260614-1601_vfull1"
    checkpoint = (
        f"/content/food101-runs/{run_name}/checkpoints/{run_name}_best_model.pt"
    )

    assert infer_run_name_from_path(checkpoint) == run_name


def test_infer_run_name_from_standard_artifact_filename() -> None:
    run_name = "baseline_cnn_simple_20260614-1601_vfull1"

    assert infer_run_name_from_path(f"/tmp/{run_name}_best_model.pt") == run_name
    assert infer_run_name_from_path(f"/tmp/{run_name}_model.pt") == run_name


def test_latest_run_for_model_uses_filename_timestamp(tmp_path: Path) -> None:
    config = load_config("configs/resnet50.yaml")
    older = create_run_paths(tmp_path, build_run_name(config, timestamp="20260601-1200"))
    newer = create_run_paths(tmp_path, build_run_name(config, timestamp="20260601-1300"))

    write_run_manifest(older, config=config, stage="training", status="complete")
    write_run_manifest(newer, config=config, stage="training", status="complete")

    latest = latest_run_for_model(tmp_path, "resnet50")
    assert latest is not None
    assert latest["run_name"] == newer.run_name


def test_list_incomplete_run_dirs_returns_missing_manifest_dirs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "runs"
    complete = run_root / "baseline_cnn_20260601-1200_v1"
    incomplete = run_root / "baseline_cnn_20260601-1300_v1"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    (complete / "manifest.json").write_text("{}", encoding="utf-8")

    results = list_incomplete_run_dirs(tmp_path)

    assert [item.run_dir for item in results] == [incomplete]
    assert results[0].run_name == incomplete.name
    assert results[0].reason == "missing_manifest_json"


def test_manifest_to_comparison_row() -> None:
    row = manifest_to_comparison_row(
        {
            "model_name": "efficientnet_b0",
            "model_version": "v1",
            "timestamp": "20260601-1830",
            "artifacts": {"checkpoint": "model.pt"},
            "metrics": {
                "top1": 0.7,
                "top5": 0.9,
                "macro_f1": 0.6,
                "ece": 0.05,
                "latency": {"mean_ms": 12.0},
                "params": 123,
            },
            "notes": ["candidate"],
        }
    )

    assert row["model"] == "efficientnet_b0"
    assert row["top5"] == 0.9
    assert row["latency_ms"] == 12.0


def test_resolve_manifest_artifact_rehomes_copied_absolute_run_paths(tmp_path: Path) -> None:
    config = load_config("configs/convnext_tiny.yaml")
    paths = create_run_paths(tmp_path, build_run_name(config, timestamp="20260602-1258"))
    paths.best_checkpoint_path.write_bytes(b"checkpoint")
    paths.manifest_path.write_text("{}", encoding="utf-8")

    copied_checkpoint = (
        "/tmp/remote_archive/outputs/runs/"
        f"{paths.run_name}/checkpoints/{paths.run_name}_best_model.pt"
    )
    manifest = {
        "_manifest_path": str(paths.manifest_path),
        "run_name": paths.run_name,
        "artifacts": {"checkpoint": copied_checkpoint},
    }

    assert (
        resolve_manifest_artifact(manifest, "checkpoint", project_root=tmp_path)
        == paths.best_checkpoint_path
    )
    assert manifest_with_resolved_artifacts(manifest, project_root=tmp_path)["artifacts"][
        "checkpoint"
    ] == str(paths.best_checkpoint_path)


def test_select_best_run_ranks_top5_then_macro_f1() -> None:
    best = select_best_run(
        [
            {
                "run_name": "a",
                "status": "complete",
                "timestamp": "20260601-1200",
                "metrics": {"top5": 0.8, "macro_f1": 0.7},
            },
            {
                "run_name": "b",
                "status": "complete",
                "timestamp": "20260601-1300",
                "metrics": {"top5": 0.8, "macro_f1": 0.75},
            },
            {
                "run_name": "c",
                "status": "complete",
                "timestamp": "20260601-1400",
                "metrics": {"top5": 0.7, "macro_f1": 0.9},
            },
        ]
    )

    assert best is not None
    assert best["run_name"] == "b"


def test_select_final_model_script_writes_stable_alias(tmp_path: Path) -> None:
    config = load_config("configs/efficientnet_b0.yaml")
    paths = create_run_paths(tmp_path, build_run_name(config, timestamp="20260601-1830"))
    paths.best_checkpoint_path.write_bytes(b"checkpoint")
    write_run_manifest(
        paths,
        config=config,
        stage="evaluation",
        status="complete",
        artifacts={"checkpoint": str(paths.best_checkpoint_path)},
        metrics={"top5": 0.9, "macro_f1": 0.7},
    )

    output_dir = tmp_path / "final_selected"
    subprocess.run(
        [
            sys.executable,
            "scripts/select_final_model.py",
            "--run-root",
            str(tmp_path / "outputs" / "runs"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    assert (output_dir / "final_selected_model.pt").read_bytes() == b"checkpoint"
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["source_run_name"] == paths.run_name
