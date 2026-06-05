"""Select the best completed run and write a stable final-model alias."""

from argparse import ArgumentParser
import json
import shutil
from pathlib import Path

from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.runs import (
    list_run_manifests,
    manifest_to_comparison_row,
    manifest_with_resolved_artifacts,
    resolve_manifest_artifact,
    sanitize_run_component,
    select_best_run,
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/final_selected")
    parser.add_argument("--model", action="append", default=None)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    run_root = resolve_project_path(args.run_root, project_root)
    output_dir = resolve_project_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [
        manifest_with_resolved_artifacts(manifest, project_root=project_root, run_root=run_root)
        for manifest in list_run_manifests(project_root, run_root=run_root)
    ]
    if args.model:
        allowed = {sanitize_run_component(model) for model in args.model}
        manifests = [
            manifest
            for manifest in manifests
            if sanitize_run_component(str(manifest.get("model_name", ""))) in allowed
        ]

    best = select_best_run(manifests)
    if best is None:
        raise ValueError("No completed run manifests with comparable metrics were found.")

    checkpoint = resolve_manifest_artifact(best, "checkpoint", project_root=project_root, run_root=run_root)
    if checkpoint is None or not checkpoint.is_file():
        raw_checkpoint = best.get("artifacts", {}).get("checkpoint")
        raise FileNotFoundError(f"Best run checkpoint not found: {raw_checkpoint}")

    alias_checkpoint = output_dir / "final_selected_model.pt"
    alias_manifest = output_dir / "manifest.json"
    shutil.copy2(checkpoint, alias_checkpoint)

    payload = {
        "source_run_name": best.get("run_name"),
        "source_manifest": best.get("_manifest_path"),
        "selection_rule": "max top5, then max macro_f1",
        "checkpoint": str(alias_checkpoint),
        "comparison_row": manifest_to_comparison_row(best),
    }
    alias_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"source_run={best.get('run_name')}")
    print(f"final_checkpoint={alias_checkpoint}")
    print(f"final_manifest={alias_manifest}")


if __name__ == "__main__":
    main()
