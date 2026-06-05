"""Export a trained Food-101 model to PyTorch and optionally ONNX."""

from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-pt", default=None)
    parser.add_argument("--output-onnx", default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--skip-pt", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--require-onnx", action="store_true")
    parser.add_argument("--opset-version", type=int, default=17)
    return parser


def main() -> None:
    args = parse_args().parse_args()

    from food101_cnn.inference.export import (
        export_onnx_model,
        export_pytorch_state,
        load_model_checkpoint,
    )
    from food101_cnn.inference.predict import load_class_names
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
    run_name = (
        args.run_name
        or infer_run_name_from_path(checkpoint_path)
        or build_run_name(config)
    )
    run_paths = create_run_paths(project_root, run_name, run_root=args.run_root)
    output_pt = (
        resolve_project_path(args.output_pt, project_root)
        if args.output_pt
        else run_paths.exports_dir / f"{run_name}_model.pt"
    )
    output_onnx = (
        resolve_project_path(args.output_onnx, project_root)
        if args.output_onnx
        else run_paths.exports_dir / f"{run_name}_model.onnx"
    )

    model = build_model_from_config(config)
    load_model_checkpoint(model, checkpoint_path)
    class_names = load_class_names(
        explicit_path=resolve_project_path(args.class_names, project_root)
        if args.class_names
        else None,
        config=config,
        checkpoint_path=checkpoint_path,
        num_classes=int(config["model"]["num_classes"]),
    )

    artifacts = {"checkpoint": str(checkpoint_path)}
    notes: list[str] = []
    if not args.skip_pt:
        exported_pt = export_pytorch_state(
            model,
            output_pt,
            config=config,
            class_names=class_names,
            extra_state={"source_checkpoint": str(checkpoint_path)},
        )
        artifacts["pt_export"] = str(exported_pt)
        print(f"pt_export={exported_pt}")

    onnx_status = "skipped"
    if not args.skip_onnx and bool(config["inference"].get("export_onnx", True)):
        try:
            exported_onnx = export_onnx_model(
                model,
                output_onnx,
                input_shape=(1, 3, int(config["data"]["image_size"]), int(config["data"]["image_size"])),
                device=select_device(str(config["training"].get("device", "auto"))),
                opset_version=args.opset_version,
            )
            artifacts["onnx_export"] = str(exported_onnx)
            onnx_status = "complete"
            print(f"onnx_export={exported_onnx}")
        except Exception as exc:
            if args.require_onnx:
                raise
            onnx_status = "skipped_missing_dependency_or_export_error"
            artifacts["onnx_export"] = None
            notes.append(f"onnx_export_failed: {exc}")
            print(f"onnx_export=failed: {exc}")

    write_run_manifest(
        run_paths,
        config=config,
        stage="export",
        status="complete",
        artifacts=artifacts,
        metrics={"onnx_status": onnx_status},
        notes=[*notes, "export attempted via scripts/export_model.py"],
    )
    print(f"run_name={run_name}")
    print(f"run_dir={run_paths.run_dir}")


if __name__ == "__main__":
    main()
