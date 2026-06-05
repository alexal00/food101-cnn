import subprocess
import sys
from pathlib import Path


def test_clean_runs_lists_incomplete_dirs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    incomplete = run_root / "baseline_cnn_20260601-1200_v1"
    complete = run_root / "efficientnet_b0_20260601-1300_v1"
    incomplete.mkdir(parents=True)
    complete.mkdir(parents=True)
    (complete / "manifest.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/clean_runs.py",
            "--run-root",
            str(run_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "incomplete_count=1" in result.stdout
    assert str(incomplete) in result.stdout
    assert str(complete) not in result.stdout
    assert incomplete.is_dir()


def test_clean_runs_refuses_delete_without_yes(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    incomplete = run_root / "baseline_cnn_20260601-1200_v1"
    incomplete.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/clean_runs.py",
            "--run-root",
            str(run_root),
            "--delete",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to delete without --yes." in result.stderr
    assert incomplete.is_dir()


def test_clean_runs_deletes_only_with_delete_and_yes(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    incomplete = run_root / "baseline_cnn_20260601-1200_v1"
    complete = run_root / "efficientnet_b0_20260601-1300_v1"
    incomplete.mkdir(parents=True)
    complete.mkdir(parents=True)
    (complete / "manifest.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/clean_runs.py",
            "--run-root",
            str(run_root),
            "--delete",
            "--yes",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "deleted_run=" in result.stdout
    assert not incomplete.exists()
    assert complete.is_dir()
