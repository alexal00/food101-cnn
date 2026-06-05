"""Run single-image or folder inference with top-k probabilities."""

from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", default="outputs/runs")
    return parser


def main() -> None:
    args = parse_args().parse_args()

    from food101_cnn.data.transforms import build_eval_transform
    from food101_cnn.inference.predict import (
        iter_input_images,
        load_class_names,
        load_checkpoint_state,
        predict_paths,
        save_predictions_csv,
    )
    from food101_cnn.models.registry import build_model_from_config
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
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)
    input_path = resolve_project_path(args.input, project_root)
    run_name = (
        args.run_name
        or infer_run_name_from_path(checkpoint_path)
        or build_run_name(config)
    )
    run_paths = create_run_paths(project_root, run_name, run_root=args.run_root)
    top_k = args.top_k or int(config["evaluation"]["top_k"])
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(config["inference"]["threshold"])
    )

    model = build_model_from_config(config)
    model.load_state_dict(load_checkpoint_state(checkpoint_path))
    class_names = load_class_names(
        explicit_path=resolve_project_path(args.class_names, project_root)
        if args.class_names
        else None,
        config=config,
        checkpoint_path=checkpoint_path,
        num_classes=int(config["model"]["num_classes"]),
    )
    device = select_device(str(config["training"].get("device", "auto")))
    image_paths = iter_input_images(input_path)

    predictions = predict_paths(
        model,
        image_paths,
        transform=build_eval_transform(config),
        class_names=class_names,
        device=device,
        top_k=top_k,
        threshold=threshold,
    )

    csv_output = None
    if not args.no_csv:
        csv_output = (
            resolve_project_path(args.csv_output, project_root)
            if args.csv_output
            else run_paths.predictions_dir / f"{run_name}_predictions.csv"
        )
        save_predictions_csv(predictions, csv_output)
        print(f"csv_output={csv_output}")
        write_run_manifest(
            run_paths,
            config=config,
            stage="prediction",
            status="complete",
            artifacts={
                "checkpoint": str(checkpoint_path),
                "predictions_csv": str(csv_output),
            },
            metrics={"prediction_count": len(predictions)},
            notes=["predictions generated via scripts/predict.py"],
        )

    print(f"run_name={run_name}")
    for prediction in predictions:
        top_items = ", ".join(
            f"{label}:{probability:.4f}"
            for label, probability in zip(
                prediction.top_k_labels,
                prediction.top_k_probabilities,
                strict=True,
            )
        )
        print(
            f"{prediction.image_path}\tdecision={prediction.decision}"
            f"\tpredicted={prediction.predicted_label}"
            f"\tconfidence={prediction.confidence:.4f}\ttop_k={top_items}"
        )


if __name__ == "__main__":
    main()
