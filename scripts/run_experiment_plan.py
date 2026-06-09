"""Run a staged Food-101 experiment plan through the stable CLI scripts."""

from argparse import ArgumentParser
from pathlib import Path
import subprocess
import sys

from food101_cnn.utils.experiment_plan import (
    ExperimentPlanItem,
    filter_training_plan,
    get_training_plan,
    plan_batch_size,
    selected_index_csv,
)
from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.runs import latest_run_for_model


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="gpu-default", choices=["gpu-default", "gpu-full"])
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--data-config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--index-csv", default="data/reports/dataset_index.csv")
    parser.add_argument("--cache-root", default="data/processed")
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--gpu-memory-gb", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume-checkpoint", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-prep", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--export-pt", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    index_csv = resolve_project_path(args.index_csv, project_root)
    cache_root = resolve_project_path(args.cache_root, project_root)
    run_root = resolve_project_path(args.run_root, project_root)
    memory_gb = args.gpu_memory_gb if args.gpu_memory_gb is not None else detect_gpu_memory_gb()
    plan = filter_training_plan(get_training_plan(args.plan), args.model)
    resume_checkpoints = parse_resume_checkpoints(args.resume_checkpoint)

    if not plan:
        raise ValueError("Selected plan is empty.")

    requested_actions = any(
        [args.data_prep, args.cache_images, args.train, args.evaluate, args.export_pt]
    )
    execute = requested_actions and not args.dry_run

    if args.data_prep:
        for command in data_prep_commands(args.python, args.data_config, index_csv, project_root):
            run_command(command, cwd=project_root, execute=execute)
    if args.cache_images:
        run_command(
            cache_command(args.python, args.data_config, index_csv, cache_root),
            cwd=project_root,
            execute=execute,
        )

    active_index_csv = selected_index_csv(index_csv, cache_root)

    if args.train:
        for item in plan:
            command = train_command(
                args.python,
                item,
                index_csv=active_index_csv,
                run_root=run_root,
                memory_gb=memory_gb,
                device=args.device,
                num_workers=args.num_workers,
                resume_checkpoint=resume_checkpoints.get(item.model),
                max_batches=args.max_batches,
                log_level=args.log_level,
            )
            run_command(command, cwd=project_root, execute=execute)

    if args.evaluate:
        for item in plan:
            command = evaluate_command(
                args.python,
                item,
                project_root=project_root,
                index_csv=active_index_csv,
                run_root=run_root,
                memory_gb=memory_gb,
                device=args.device,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
            )
            if command is not None:
                run_command(command, cwd=project_root, execute=execute)

    if args.export_pt:
        for item in plan:
            command = export_command(
                args.python,
                item,
                project_root=project_root,
                run_root=run_root,
                export_onnx=args.export_onnx,
            )
            if command is not None:
                run_command(command, cwd=project_root, execute=execute)

    if not requested_actions:
        print_plan(plan, memory_gb=memory_gb)


def detect_gpu_memory_gb() -> float:
    """Return detected CUDA memory in GB, or 0 when CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return 0.0
    return 0.0


def parse_resume_checkpoints(values: list[str]) -> dict[str, str]:
    """Parse ``MODEL=PATH`` resume-checkpoint options."""
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Resume checkpoint must use MODEL=PATH format: {value}")
        model, checkpoint = value.split("=", 1)
        parsed[model] = checkpoint
    return parsed


def data_prep_commands(
    python: str,
    data_config: str,
    index_csv: Path,
    project_root: Path,
) -> list[list[str]]:
    """Build data download/index and validation commands."""
    reports_dir = project_root / "data" / "reports"
    return [
        [
            python,
            "scripts/download_data.py",
            "--config",
            data_config,
            "--index-output",
            str(index_csv),
        ],
        [
            python,
            "scripts/validate_images.py",
            "--config",
            data_config,
            "--index-csv",
            str(index_csv),
            "--report-csv",
            str(reports_dir / "image_validation_report.csv"),
            "--corrupted-output",
            str(reports_dir / "corrupted_images.txt"),
        ],
    ]


def cache_command(
    python: str,
    data_config: str,
    index_csv: Path,
    cache_root: Path,
) -> list[str]:
    """Build the resized-image cache command."""
    return [
        python,
        "scripts/cache_images.py",
        "--config",
        data_config,
        "--index-csv",
        str(index_csv),
        "--cache-root",
        str(cache_root),
    ]


def train_command(
    python: str,
    item: ExperimentPlanItem,
    *,
    index_csv: Path,
    run_root: Path,
    memory_gb: float,
    device: str | None,
    num_workers: int | None,
    resume_checkpoint: str | None,
    max_batches: int | None,
    log_level: str,
) -> list[str]:
    """Build a model training command."""
    command = [
        python,
        "scripts/train.py",
        "--config",
        item.config,
        "--index-csv",
        str(index_csv),
        "--epochs",
        str(item.epochs),
        "--batch-size",
        str(plan_batch_size(item, memory_gb)),
        "--num-workers",
        str(num_workers if num_workers is not None else item.num_workers),
        "--run-root",
        str(run_root),
        "--log-level",
        log_level,
    ]
    if device is not None:
        command.extend(["--device", device])
    command.append("--mixed-precision" if item.mixed_precision else "--no-mixed-precision")
    if resume_checkpoint:
        command.extend(["--resume-checkpoint", resume_checkpoint])
    if max_batches is not None:
        command.extend(["--max-batches", str(max_batches)])
    return command


def evaluate_command(
    python: str,
    item: ExperimentPlanItem,
    *,
    project_root: Path,
    index_csv: Path,
    run_root: Path,
    memory_gb: float,
    device: str | None,
    num_workers: int | None,
    max_batches: int | None,
) -> list[str] | None:
    """Build an evaluation command for the latest completed run of a model."""
    manifest = latest_run_for_model(project_root, item.model, run_root=run_root)
    if not manifest:
        print(f"No complete run manifest found for {item.model}")
        return None
    checkpoint = manifest.get("artifacts", {}).get("checkpoint")
    if not checkpoint:
        print(f"No checkpoint recorded for {item.model}")
        return None

    command = [
        python,
        "scripts/evaluate.py",
        "--config",
        item.config,
        "--checkpoint",
        str(checkpoint),
        "--index-csv",
        str(index_csv),
        "--split",
        "test",
        "--batch-size",
        str(plan_batch_size(item, memory_gb)),
        "--num-workers",
        str(num_workers if num_workers is not None else item.num_workers),
        "--run-root",
        str(run_root),
    ]
    if device is not None:
        command.extend(["--device", device])
    if max_batches is not None:
        command.extend(["--max-batches", str(max_batches)])
    return command


def export_command(
    python: str,
    item: ExperimentPlanItem,
    *,
    project_root: Path,
    run_root: Path,
    export_onnx: bool,
) -> list[str] | None:
    """Build a PyTorch export command for the latest completed run of a model."""
    manifest = latest_run_for_model(project_root, item.model, run_root=run_root)
    if not manifest:
        print(f"No complete run manifest found for {item.model}")
        return None
    checkpoint = manifest.get("artifacts", {}).get("checkpoint")
    if not checkpoint:
        print(f"No checkpoint recorded for {item.model}")
        return None

    command = [
        python,
        "scripts/export_model.py",
        "--config",
        item.config,
        "--checkpoint",
        str(checkpoint),
        "--run-root",
        str(run_root),
    ]
    if not export_onnx:
        command.append("--skip-onnx")
    return command


def run_command(command: list[str], *, cwd: Path, execute: bool) -> None:
    """Print and optionally execute a command."""
    print("$", " ".join(command))
    if execute:
        subprocess.run(command, cwd=cwd, check=True)


def print_plan(items: tuple[ExperimentPlanItem, ...], *, memory_gb: float) -> None:
    """Print the selected plan without running commands."""
    for item in items:
        print(
            f"{item.model}: config={item.config} epochs={item.epochs} "
            f"batch_size={plan_batch_size(item, memory_gb)} mixed_precision={item.mixed_precision}"
        )


if __name__ == "__main__":
    main()
