# Food-101 Run Logbook

This logbook is a lightweight committed index of finished runs. It is generated
from `outputs/runs/*/manifest.json` and intentionally excludes checkpoints,
TensorBoard logs, cached images, generated figures, and incomplete run
directories without `manifest.json`.

Default scope: latest completed run per model.

| Model | Run | Date | Config | Stage | Top-1 | Top-5 | Macro F1 | ECE | Checkpoint | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline_cnn | baseline_cnn_20260602-0000_v1 | 20260602-0000 | baseline_cnn v1 | export | 0.6593 | 0.8888 | 0.6616 | 0.1347 | `outputs/runs/baseline_cnn_20260602-0000_v1/checkpoints/baseline_cnn_20260602-0000_v1_best_model.pt` | training completed via scripts/train.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py |
| convnext_tiny | convnext_tiny_20260602-1258_v1 | 20260602-1258 | convnext_tiny v1 | export | 0.7431 | 0.9266 | 0.7417 | 0.1847 | `outputs/runs/convnext_tiny_20260602-1258_v1/checkpoints/convnext_tiny_20260602-1258_v1_best_model.pt` | training completed via scripts/train.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py; onnx_export_failed: No module named 'onnxscript'; export attempted via scripts/export_model.py; onnx_export_failed: No module named 'onnxscript'; export attempted via scripts/export_model.py; onnx_export_failed: No module named 'onnxscript'; export attempted via scripts/export_model.py |
| efficientnet_b0 | efficientnet_b0_20260602-0040_v1 | 20260602-0040 | efficientnet_b0 v1 | export | 0.5843 | 0.8205 | 0.5771 | 0.1598 | `outputs/runs/efficientnet_b0_20260602-0040_v1/checkpoints/efficientnet_b0_20260602-0040_v1_best_model.pt` | training completed via scripts/train.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py |
| resnet50 | resnet50_20260602-0657_v1 | 20260602-0657 | resnet50 v1 | export | 0.6597 | 0.8766 | 0.6557 | 0.1947 | `outputs/runs/resnet50_20260602-0657_v1/checkpoints/resnet50_20260602-0657_v1_best_model.pt` | training completed via scripts/train.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py; evaluated split=test via scripts/evaluate.py; export attempted via scripts/export_model.py |
