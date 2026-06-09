"""Evaluate a trained Food-101 checkpoint and write reports."""

from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-csv", default="data/reports/dataset_index.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--misclassified-output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def main() -> None:
    args = parse_args().parse_args()

    import json
    from dataclasses import asdict

    import torch
    from torch.utils.data import DataLoader

    from food101_cnn.data.dataset import Food101IndexedDataset
    from food101_cnn.data.transforms import build_eval_transform
    from food101_cnn.evaluation.calibration import (
        calibration_report,
        save_calibration_report,
    )
    from food101_cnn.evaluation.confusion import (
        compute_confusion_matrix,
        save_confusion_matrix_csv,
    )
    from food101_cnn.evaluation.errors import (
        collect_misclassifications,
        prediction_records_from_logits,
        save_prediction_records_csv,
        save_misclassifications_csv,
    )
    from food101_cnn.evaluation.latency import measure_inference_latency
    from food101_cnn.evaluation.metrics import classification_metrics_from_logits
    from food101_cnn.models.registry import build_model_from_config
    from food101_cnn.training.losses import build_classification_loss
    from food101_cnn.utils.device import select_device
    from food101_cnn.utils.runs import (
        build_run_name,
        create_run_paths,
        infer_run_name_from_path,
        write_run_manifest,
    )

    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)
    if args.device is not None:
        config["training"]["device"] = args.device
    index_csv = resolve_project_path(args.index_csv, project_root)
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)
    run_name = (
        args.run_name
        or infer_run_name_from_path(checkpoint_path)
        or build_run_name(config)
    )
    run_paths = create_run_paths(project_root, run_name, run_root=args.run_root)
    output_dir = (
        resolve_project_path(args.output_dir, project_root)
        if args.output_dir
        else run_paths.reports_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(str(config["training"].get("device", "auto")))

    dataset = Food101IndexedDataset(
        index_csv,
        split=args.split,
        transform=build_eval_transform(config),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        **_loader_kwargs(args.num_workers, pin_memory=device.type == "cuda"),
    )

    model = build_model_from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    criterion = build_classification_loss(
        label_smoothing=float(config["training"].get("label_smoothing", 0.0))
    )

    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    image_paths: list[str] = []
    loss_sum = 0.0
    sample_count = 0
    latency_batch: torch.Tensor | None = None
    cursor = 0

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break

            if latency_batch is None:
                latency_batch = inputs[:1].clone()

            batch_size = targets.size(0)
            batch_paths = [
                str(record.image_path)
                for record in dataset.records[cursor : cursor + batch_size]
            ]
            cursor += batch_size

            non_blocking = device.type == "cuda"
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            logits = model(inputs)
            loss = criterion(logits, targets)

            logits_list.append(logits.cpu())
            targets_list.append(targets.cpu())
            image_paths.extend(batch_paths)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("No evaluation samples were processed.")

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    top_k = args.top_k or int(config["evaluation"]["top_k"])
    class_names = _class_names_from_dataset(dataset)

    metrics = classification_metrics_from_logits(
        logits,
        targets,
        num_classes=len(class_names),
        top_k=top_k,
    )
    confusion = compute_confusion_matrix(
        logits.argmax(dim=1),
        targets,
        num_classes=len(class_names),
    )
    calibration = calibration_report(
        logits,
        targets,
        num_bins=args.calibration_bins,
    )
    misclassifications = collect_misclassifications(
        logits,
        targets,
        class_names=class_names,
        image_paths=image_paths,
        top_k=top_k,
    )
    prediction_records = prediction_records_from_logits(
        logits,
        targets,
        class_names=class_names,
        image_paths=image_paths,
        top_k=top_k,
    )

    metrics_output = output_dir / "evaluation_metrics.json"
    confusion_output = output_dir / "confusion_matrix.csv"
    calibration_output = output_dir / "calibration_report.json"
    predictions_output = output_dir / f"predictions_{args.split}.csv"
    misclassified_output = (
        resolve_project_path(args.misclassified_output, project_root)
        if args.misclassified_output
        else run_paths.reports_dir / f"{run_name}_misclassified.csv"
    )

    save_confusion_matrix_csv(confusion, class_names, confusion_output)
    save_calibration_report(calibration, calibration_output)
    save_prediction_records_csv(prediction_records, predictions_output)
    save_misclassifications_csv(misclassifications, misclassified_output)

    payload = {
        **asdict(metrics),
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "split": args.split,
        "top_k": top_k,
        "ece": calibration.ece,
    }

    if latency_batch is not None and bool(config["evaluation"].get("measure_latency", True)):
        latency = measure_inference_latency(
            model,
            latency_batch,
            device,
            repeats=args.latency_repeats,
        )
        payload["latency"] = asdict(latency)

    metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_run_manifest(
        run_paths,
        config=config,
        stage="evaluation",
        status="complete",
        artifacts={
            "checkpoint": str(checkpoint_path),
            "metrics_json": str(metrics_output),
            "confusion_csv": str(confusion_output),
            "calibration_json": str(calibration_output),
            "predictions_csv": str(predictions_output),
            "misclassified_csv": str(misclassified_output),
        },
        metrics=payload,
        notes=[f"evaluated split={args.split} via scripts/evaluate.py"],
    )

    print(f"run_name={run_name}")
    print(f"run_dir={run_paths.run_dir}")
    print(f"split={args.split}")
    print(f"samples={sample_count}")
    print(f"loss={payload['loss']:.6f}")
    print(f"top1={metrics.top1:.6f}")
    print(f"top{top_k}={metrics.top5:.6f}")
    print(f"macro_f1={metrics.macro_f1:.6f}")
    print(f"ece={calibration.ece:.6f}")
    print(f"metrics_json={metrics_output}")
    print(f"confusion_csv={confusion_output}")
    print(f"calibration_json={calibration_output}")
    print(f"predictions_csv={predictions_output}")
    print(f"misclassified_csv={misclassified_output}")


def _class_names_from_dataset(dataset) -> list[str]:
    mapping = {record.label_index: record.label for record in dataset.records}
    return [mapping[index] for index in sorted(mapping)]


def _loader_kwargs(num_workers: int, *, pin_memory: bool) -> dict[str, object]:
    """Return DataLoader options tuned for GPU evaluation throughput."""
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


if __name__ == "__main__":
    main()
