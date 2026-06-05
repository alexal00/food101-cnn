import subprocess
import sys
from pathlib import Path

from food101_cnn.utils.run_logbook import (
    build_run_logbook_entries,
    render_run_logbook_markdown,
)
from food101_cnn.utils.runs import build_run_name, create_run_paths, write_run_manifest


def _config(model_name: str) -> dict:
    return {"model": {"name": model_name, "version": "v1"}}


def _write_manifest(
    project_root: Path,
    model_name: str,
    timestamp: str,
    *,
    status: str = "complete",
    top5: float = 0.8,
) -> None:
    config = _config(model_name)
    paths = create_run_paths(project_root, build_run_name(config, timestamp=timestamp))
    paths.best_checkpoint_path.write_bytes(b"checkpoint")
    write_run_manifest(
        paths,
        config=config,
        stage="export",
        status=status,
        artifacts={"checkpoint": str(paths.best_checkpoint_path)},
        metrics={"top1": 0.7, "top5": top5, "macro_f1": 0.6, "ece": 0.05},
        notes=["kept for comparison"],
    )


def test_build_run_logbook_entries_keeps_latest_completed_run_per_model(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "efficientnet_b0", "20260601-1200", top5=0.81)
    _write_manifest(tmp_path, "efficientnet_b0", "20260601-1300", top5=0.82)
    _write_manifest(tmp_path, "resnet50", "20260601-1230", top5=0.83)
    _write_manifest(tmp_path, "convnext_tiny", "20260601-1400", status="running")
    (tmp_path / "outputs" / "runs" / "baseline_cnn_20260601-1500_v1").mkdir(
        parents=True
    )

    entries = build_run_logbook_entries(tmp_path)

    assert [entry.run_name for entry in entries] == [
        "efficientnet_b0_20260601-1300_v1",
        "resnet50_20260601-1230_v1",
    ]
    assert entries[0].checkpoint.endswith(
        "outputs/runs/efficientnet_b0_20260601-1300_v1/checkpoints/"
        "efficientnet_b0_20260601-1300_v1_best_model.pt"
    )


def test_render_run_logbook_markdown_formats_metrics_and_notes(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "baseline_cnn", "20260601-1200", top5=0.87654)

    markdown = render_run_logbook_markdown(build_run_logbook_entries(tmp_path))

    assert "# Food-101 Run Logbook" in markdown
    assert "baseline_cnn_20260601-1200_v1" in markdown
    assert "| baseline_cnn |" in markdown
    assert "| 0.7000 | 0.8765 | 0.6000 | 0.0500 |" in markdown
    assert "kept for comparison" in markdown


def test_update_run_logbook_script_writes_markdown(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "resnet50", "20260601-1200")
    output_path = tmp_path / "run_logbook.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/update_run_logbook.py",
            "--run-root",
            str(tmp_path / "outputs" / "runs"),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "logbook_entries=1" in result.stdout
    assert output_path.is_file()
    assert "resnet50_20260601-1200_v1" in output_path.read_text(encoding="utf-8")
