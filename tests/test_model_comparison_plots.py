from pathlib import Path

import pandas as pd

from food101_cnn.evaluation.model_comparison import (
    build_model_comparison_table,
    latest_manifest_per_model,
    save_global_model_comparison_plots,
)


def test_model_comparison_table_and_plots(tmp_path: Path) -> None:
    manifests = [
        {
            "model_name": "baseline_cnn",
            "model_version": "v1",
            "timestamp": "20260601-1000",
            "artifacts": {"checkpoint": "a.pt"},
            "metrics": {"top1": 0.4, "top5": 0.7, "macro_f1": 0.35},
            "notes": [],
        },
        {
            "model_name": "efficientnet_b0",
            "model_version": "v1",
            "timestamp": "20260601-1100",
            "artifacts": {"checkpoint": "b.pt"},
            "metrics": {
                "top1": 0.6,
                "top5": 0.9,
                "macro_f1": 0.55,
                "latency": {"mean_ms": 3.2},
            },
            "notes": [],
        },
    ]

    latest = latest_manifest_per_model(manifests)
    table = build_model_comparison_table(latest)
    outputs = save_global_model_comparison_plots(table, tmp_path)

    assert list(table["model"]) == ["efficientnet_b0", "baseline_cnn"]
    assert isinstance(table, pd.DataFrame)
    assert (tmp_path / "model_top5_accuracy.png").is_file()
    assert outputs["summary_table"].is_file()
