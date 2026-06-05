"""Generate post-processing reports and figures from saved Food-101 run artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from food101_cnn.config import load_config
from food101_cnn.evaluation.calibration import (
    load_calibration_report_json,
    save_calibration_artifacts_from_report,
    save_calibration_plots_from_predictions,
)
from food101_cnn.evaluation.class_report import (
    build_classification_report_from_confusion,
    build_classification_report_table,
    save_class_report_tables,
    top_best_classes,
    top_worst_classes,
)
from food101_cnn.evaluation.confusion import (
    compute_confusion_matrix,
    load_confusion_matrix_csv,
    save_confusion_analysis,
)
from food101_cnn.evaluation.errors import (
    normalize_prediction_dataframe,
    save_error_galleries,
    save_misclassified_examples_csv,
)
from food101_cnn.evaluation.model_comparison import (
    build_model_comparison_table,
    discover_available_runs,
    latest_manifest_per_model,
    save_global_model_comparison_plots,
)
from food101_cnn.evaluation.plot_training import (
    analyze_training_diagnostics,
    save_training_curve_plots,
    save_training_diagnostics,
)
from food101_cnn.evaluation.tensorboard_reader import (
    load_tensorboard_scalars,
    save_tensorboard_scalars,
)
from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.runs import latest_run_for_model, manifest_with_resolved_artifacts


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--log-dir", default="outputs/tensorboard")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--confusion-csv", default=None)
    parser.add_argument("--misclassified-csv", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--calibration-bins", type=int, default=15)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)
    output_dir = resolve_project_path(args.output_dir, project_root)
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    manifest = _latest_manifest_for_config(project_root, config)
    artifacts = manifest.get("artifacts", {}) if manifest else {}

    log_dir = resolve_project_path(args.log_dir, project_root)
    scalars = load_tensorboard_scalars(log_dir)
    if scalars.empty and (output_dir / "runs").is_dir() and (output_dir / "runs") != log_dir:
        scalars = load_tensorboard_scalars(output_dir / "runs")
    generated.append(save_tensorboard_scalars(scalars, reports_dir / "tensorboard_scalars.csv"))
    generated.extend(save_training_curve_plots(scalars, figures_dir / "training_curves").values())
    diagnostics = analyze_training_diagnostics(scalars)
    generated.extend(save_training_diagnostics(diagnostics, reports_dir).values())

    comparison_manifests = latest_manifest_per_model(discover_available_runs(project_root))
    comparison_table = build_model_comparison_table(comparison_manifests)
    if not comparison_table.empty:
        generated.extend(
            save_global_model_comparison_plots(
                comparison_table,
                figures_dir / "model_comparison",
            ).values()
        )

    predictions_path = _resolve_optional_path(
        args.predictions,
        project_root,
        artifacts.get("predictions_csv"),
    )
    confusion_path = _resolve_optional_path(
        args.confusion_csv,
        project_root,
        artifacts.get("confusion_csv"),
    )
    misclassified_path = _resolve_optional_path(
        args.misclassified_csv,
        project_root,
        artifacts.get("misclassified_csv"),
    )
    calibration_path = _resolve_optional_path(
        args.calibration_json,
        project_root,
        artifacts.get("calibration_json"),
    )

    class_report = None
    matrix = None
    class_names: list[str] = []
    prediction_frame = None

    if predictions_path is not None and predictions_path.is_file():
        prediction_frame = normalize_prediction_dataframe(pd.read_csv(predictions_path))
        copied_predictions = reports_dir / predictions_path.name
        prediction_frame.to_csv(copied_predictions, index=False)
        generated.append(copied_predictions)
        y_true, y_pred, class_names = _labels_from_predictions(prediction_frame)
        class_report = build_classification_report_table(y_true, y_pred, class_names)
        matrix = compute_confusion_matrix(
            torch.as_tensor(y_pred),
            torch.as_tensor(y_true),
            num_classes=len(class_names),
        )
    elif confusion_path is not None and confusion_path.is_file():
        matrix, class_names = load_confusion_matrix_csv(confusion_path)
        class_report = build_classification_report_from_confusion(matrix, class_names)

    if class_report is not None and matrix is not None:
        generated.extend(save_class_report_tables(class_report, reports_dir).values())
        best_classes = top_best_classes(class_report)["class_name"].astype(str).tolist()
        worst_classes = top_worst_classes(class_report)["class_name"].astype(str).tolist()
        generated.extend(
            save_confusion_analysis(
                matrix,
                class_names,
                figures_dir / "confusion",
                best_classes=best_classes,
                worst_classes=worst_classes,
                top_n=args.top_n,
            ).values()
        )

    if prediction_frame is not None:
        generated.append(save_misclassified_examples_csv(prediction_frame, reports_dir / "misclassified_examples.csv"))
        generated.extend(
            save_error_galleries(
                prediction_frame,
                figures_dir / "errors",
                class_report=class_report,
                project_root=project_root,
            ).values()
        )
        generated.extend(
            save_calibration_plots_from_predictions(
                prediction_frame,
                figures_dir / "calibration",
                reports_dir,
                num_bins=args.calibration_bins,
            ).values()
        )
    else:
        if misclassified_path is not None and misclassified_path.is_file():
            misclassified_frame = normalize_prediction_dataframe(pd.read_csv(misclassified_path))
            generated.append(
                save_misclassified_examples_csv(
                    misclassified_frame,
                    reports_dir / "misclassified_examples.csv",
                )
            )
            generated.extend(
                save_error_galleries(
                    misclassified_frame,
                    figures_dir / "errors",
                    class_report=class_report,
                    project_root=project_root,
                ).values()
            )
        if calibration_path is not None and calibration_path.is_file():
            report = load_calibration_report_json(calibration_path)
            generated.extend(
                save_calibration_artifacts_from_report(
                    report,
                    figures_dir / "calibration",
                    reports_dir,
                ).values()
            )

    print("Generated post-processing artifacts:")
    for path in dict.fromkeys(generated):
        print(f"- {path}")
    print(f"diagnostic_status={diagnostics['status']}")


def _latest_manifest_for_config(project_root: Path, config: dict) -> dict | None:
    model_name = str(config.get("model", {}).get("name", ""))
    if not model_name:
        return None
    manifest = latest_run_for_model(project_root, model_name)
    if manifest is None:
        return None
    return manifest_with_resolved_artifacts(manifest, project_root=project_root)


def _resolve_optional_path(
    explicit: str | None,
    project_root: Path,
    manifest_value: str | None,
) -> Path | None:
    raw = explicit or manifest_value
    if not raw:
        return None
    return resolve_project_path(raw, project_root)


def _labels_from_predictions(frame: pd.DataFrame) -> tuple[list[int], list[int], list[str]]:
    if {"true_idx", "pred_idx"}.issubset(frame.columns):
        index_name_pairs = pd.concat(
            [
                frame[["true_idx", "true_class"]].rename(columns={"true_idx": "idx", "true_class": "class_name"}),
                frame[["pred_idx", "pred_class"]].rename(columns={"pred_idx": "idx", "pred_class": "class_name"}),
            ],
            ignore_index=True,
        )
        index_name_pairs["idx"] = pd.to_numeric(index_name_pairs["idx"], errors="coerce")
        index_name_pairs = index_name_pairs.dropna(subset=["idx"])
        mapping = {
            int(row.idx): str(row.class_name)
            for row in index_name_pairs.sort_values("idx").itertuples(index=False)
        }
        class_names = [mapping[index] for index in sorted(mapping)]
        return (
            frame["true_idx"].astype(int).tolist(),
            frame["pred_idx"].astype(int).tolist(),
            class_names,
        )

    class_names = sorted(set(frame["true_class"].astype(str)).union(frame["pred_class"].astype(str)))
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return (
        frame["true_class"].astype(str).map(class_to_index).astype(int).tolist(),
        frame["pred_class"].astype(str).map(class_to_index).astype(int).tolist(),
        class_names,
    )


if __name__ == "__main__":
    main()
