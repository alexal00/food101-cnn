"""Train a Food-101 model stage from a generated dataset index."""

import logging
from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.utils.logging import configure_logging
from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.runs import build_run_name, create_run_paths, write_run_manifest

LOGGER = logging.getLogger(__name__)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--index-csv", default="data/reports/dataset_index.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--mixed-precision", action="store_true", default=None)
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--quiet", action="store_true", help="Disable per-epoch progress logs.")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    configure_logging("WARNING" if args.quiet else args.log_level)

    from torch.utils.data import DataLoader

    from food101_cnn.data.dataset import Food101IndexedDataset
    from food101_cnn.data.transforms import build_eval_transform, build_train_transform
    from food101_cnn.models.registry import build_model_from_config
    from food101_cnn.training.train import fit
    from food101_cnn.utils.device import select_device
    from food101_cnn.utils.seed import seed_everything

    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.mixed_precision is not None:
        config["training"]["mixed_precision"] = args.mixed_precision
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    seed_everything(int(config["project"]["seed"]))
    device = select_device(str(config["training"].get("device", "auto")))

    index_csv = resolve_project_path(args.index_csv, project_root)
    if not index_csv.is_file():
        raise FileNotFoundError(
            f"Dataset index not found: {index_csv}. Run scripts/download_data.py first."
        )

    train_dataset = Food101IndexedDataset(
        index_csv,
        split="train",
        transform=build_train_transform(config),
    )
    val_dataset = Food101IndexedDataset(
        index_csv,
        split="val",
        transform=build_eval_transform(config),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(config)
    epochs = args.epochs or int(config["training"].get("epochs_head", 1))
    lr = args.lr or float(config["training"].get("lr_head", 0.001))
    resume_checkpoint = (
        resolve_project_path(args.resume_checkpoint, project_root)
        if args.resume_checkpoint
        else None
    )
    from food101_cnn.utils.runs import infer_run_name_from_path

    run_name = (
        args.run_name
        or (infer_run_name_from_path(resume_checkpoint) if resume_checkpoint else None)
        or build_run_name(config)
    )
    run_paths = create_run_paths(project_root, run_name, run_root=args.run_root)
    LOGGER.info(
        "training_setup run=%s config=%s index_csv=%s model=%s version=%s train_samples=%s "
        "val_samples=%s batch_size=%s device=%s epochs=%s lr=%.6g",
        run_name,
        args.config,
        index_csv,
        config["model"]["name"],
        config["model"]["version"],
        len(train_dataset),
        len(val_dataset),
        config["training"]["batch_size"],
        device,
        epochs,
        lr,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        epochs=epochs,
        lr=lr,
        checkpoint_dir=run_paths.checkpoints_dir,
        checkpoint_filename=run_paths.best_checkpoint_path.name,
        tensorboard_dir=run_paths.tensorboard_dir,
        max_batches=args.max_batches,
        log_progress=not args.quiet,
        resume_checkpoint=resume_checkpoint,
    )

    final = history[-1]["validation"]
    final_train = history[-1]["train"]
    checkpoint_path = run_paths.best_checkpoint_path
    write_run_manifest(
        run_paths,
        config=config,
        stage="training",
        status="complete",
        artifacts={
            "checkpoint": str(checkpoint_path),
            "tensorboard_dir": str(run_paths.tensorboard_dir),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        },
        metrics={
            "train_loss": final_train.loss,
            "train_top1": final_train.top1,
            "train_top5": final_train.top5,
            "val_loss": final.loss,
            "val_top1": final.top1,
            "val_top5": final.top5,
            "params": sum(parameter.numel() for parameter in model.parameters()),
        },
        notes=["training completed via scripts/train.py"],
    )
    print(f"run_name={run_name}")
    print(f"run_dir={run_paths.run_dir}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"epochs_completed={len(history)}")
    print(f"validation_loss={final.loss:.6f}")
    print(f"validation_top1={final.top1:.6f}")
    print(f"validation_top5={final.top5:.6f}")


if __name__ == "__main__":
    main()
