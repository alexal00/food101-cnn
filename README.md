# Food-101 CNN

Modular PyTorch project for Food-101 image classification.

The final course deliverable is `notebooks/food101_CNN_final_project.ipynb`. Reusable logic lives in `src/food101_cnn/`; repeatable command-line entry points live in `scripts/`.

## Project Scope

- CNN-from-scratch baseline for pedagogical comparison.
- Transfer-learning model factory for EfficientNet-B0, ResNet-50, and ConvNeXt-Tiny.
- Official Food-101 split preservation.
- Internal validation split from the official training split only.
- Top-5 accuracy as the primary metric.
- Top-1 accuracy, macro F1, per-class accuracy, confusion matrix, calibration, misclassification analysis, inference latency, and Grad-CAM notebook section.
- Local inference with top-k probabilities and confidence thresholding.
- Native PyTorch `.pt` export and optional ONNX export.

## Repository Layout

```text
food101-cnn/
├── configs/                  # YAML experiment configuration
├── data/
│   ├── raw/                  # Food-101 original images and metadata
│   ├── processed/            # Cached resized images
│   └── reports/              # Dataset index and image validation reports
├── notebooks/
│   └── food101_CNN_final_project.ipynb
├── outputs/
│   └── runs/
│       └── model_YYYYMMDD-HHMM_version/
│           ├── checkpoints/
│           ├── exports/
│           ├── predictions/
│           ├── reports/
│           ├── tensorboard/
│           └── manifest.json
├── scripts/                  # CLI entry points
├── src/food101_cnn/          # Reusable package code
└── tests/                    # Lightweight automated tests
```

Generated datasets, checkpoints, TensorBoard logs, cached images, and reports are intentionally kept under `data/` and `outputs/`.

## Run Naming And Tracking

Every training run uses:

```text
model_name_YYYYMMDD-HHMM_version
```

Example:

```text
efficientnet_b0_20260601-1625_v1
```

The timestamp uses local Europe/Paris time. The model implementation version is set in each config under `model.version`, while the package version is stored in `food101_cnn.__version__` and copied into each run manifest.

Each run writes:

```text
outputs/runs/<run_name>/
├── checkpoints/<run_name>_best_model.pt
├── exports/<run_name>_model.pt
├── exports/<run_name>_model.onnx
├── predictions/<run_name>_predictions.csv
├── reports/evaluation_metrics.json
├── reports/confusion_matrix.csv
├── reports/calibration_report.json
├── reports/<run_name>_misclassified.csv
├── tensorboard/
└── manifest.json
```

The notebook discovers the latest run per model from filenames and manifests, then ranks candidates by top-5 accuracy followed by macro F1.

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate food101-cnn
```

For an existing environment:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest
```

On this local macOS/Conda setup, Torch imports can abort with a duplicate OpenMP runtime error unless `KMP_DUPLICATE_LIB_OK=TRUE` is set for Torch-based validation commands.

## Configuration

Main configs:

- `configs/efficientnet_b0.yaml`: recommended local transfer-learning model.
- `configs/baseline_cnn.yaml`: CNN-from-scratch baseline.
- `configs/baseline_cnn_simple.yaml`: simpler CNN architecture using the same full GPU-style baseline schedule as `baseline_cnn_gpu`.
- `configs/baseline_cnn_local.yaml`: the same simpler CNN architecture with conservative Mac/local training limits.
- `configs/resnet50.yaml`: reference transfer-learning backbone.
- `configs/convnext_tiny.yaml`: modern CNN comparison backbone.

GPU/cloud configs:

- `configs/baseline_cnn_gpu.yaml`
- `configs/efficientnet_b0_gpu.yaml`
- `configs/resnet50_gpu.yaml`
- `configs/convnext_tiny_gpu.yaml`

The GPU/cloud configs use project-relative data and output paths, CUDA as the preferred device, mixed precision, and larger batch sizes. The training notebook also trains `baseline_cnn_local` and `baseline_cnn_simple` from their main configs so those variants keep distinct run names. CLI overrides can still adjust `--batch-size`, `--device`, `--mixed-precision`, `--no-mixed-precision`, `--gradient-accumulation-steps`, and `--resume-checkpoint`.

Important fields:

- `data.root_dir`: original Food-101 data root.
- `data.processed_dir`: disk cache for resized images.
- `data.validation_fraction`: internal validation split from official train only.
- `model.name`: `baseline_cnn`, `baseline_cnn_local`, `baseline_cnn_simple`, `efficientnet_b0`, `resnet50`, or `convnext_tiny`.
- `training.batch_size`: conservative local default.
- `training.device`: `auto`, `mps`, `cuda`, or `cpu`.
- `evaluation.top_k`: top-k metric and prediction output.
- `inference.threshold`: confidence threshold for uncertain predictions.
- `logging.*`: TensorBoard, checkpoint, and figure output directories.

## Main Notebook Workflow

Open:

```bash
jupyter lab notebooks/food101_CNN_final_project.ipynb
```

The final notebook is designed as the report surface. It should be run after
data preparation, training, evaluation, and export have produced local
artifacts. Action cells are guarded by flags near the top and default to report
mode:

```python
RUN_DATA_PREP = False
RUN_TRAINING = False
RUN_FINAL_EVAL = False
RUN_POSTPROCESSING = False
RUN_INFERENCE_EXPORT = False
```

Default execution loads existing manifests/reports and performs lightweight
smoke checks only: config loading, transform shapes, model forward pass,
synthetic training loop, metrics, calibration, misclassification analysis, and
export placeholders.

Recommended notebook usage:

1. Run once with all flags `False` to verify imports and local environment.
2. Generate runs through `scripts/run_experiment_plan.py`, individual CLI scripts, or `notebooks/food101_colab_training.ipynb`.
3. Keep action flags disabled for normal final-report execution.
4. Use notebook sections 10-17 to load latest run manifests and summarize comparison outputs from `outputs/runs/`.

Notebook execution check:

```bash
KMP_DUPLICATE_LIB_OK=TRUE jupyter nbconvert \
  --to notebook \
  --execute notebooks/food101_CNN_final_project.ipynb \
  --output /tmp/food101_CNN_final_project_executed.ipynb \
  --ExecutePreprocessor.timeout=120
```

## Google Colab Pro Workflow

Use Visual Studio Code for editing locally, then run the dedicated Colab notebook for cloud GPU training:

```text
notebooks/food101_colab_training.ipynb
```

Colab cannot see files that exist only on the local machine. The training
notebook therefore clones the configured GitHub branch into
`/content/food101-cnn` each session, installs the package, and keeps data and
routine run outputs in the temporary Colab runtime. Google Drive is mounted only
to archive selected final complete runs.

Drive is not the default data source. The Colab notebook clones the repository
from GitHub, downloads or caches Food-101 inside `/content/food101-cnn/data`,
and writes temporary run artifacts under `/content/food101-runs`. Drive is used
only when explicitly archiving selected final complete runs.

The Colab notebook:

- mounts Google Drive,
- clones the configured GitHub branch into `/content/food101-cnn`,
- stores temporary data under `/content/food101-cnn/data`,
- stores temporary run artifacts under `/content/food101-runs`,
- optionally archives selected final complete runs to Drive,
- detects the available CUDA GPU,
- selects moderate batch-size overrides based on GPU memory,
- reuses the latest cached resized-image index when present,
- supports resume checkpoints,
- runs staged model plans through `scripts/run_experiment_plan.py`,
- delegates training to `scripts/train.py`,
- delegates evaluation to `scripts/evaluate.py`,
- delegates native `.pt` exports to `scripts/export_model.py --skip-onnx`.

Colab execution flags:

```python
RUN_DATA_PREP = False
RUN_IMAGE_CACHE = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_EXPORT_PT = False
RUN_ARCHIVE_FINAL_RUNS = False
```

The default plan is `gpu-default`, which trains the simple CNN GPU schedule and
the constrained local-style baseline. Use `PLAN_NAME = "gpu-full"` for the full
comparison set.

Recommended Colab order:

1. Open `notebooks/food101_colab_training.ipynb` in Colab Pro.
2. Select a GPU runtime.
3. Set `GITHUB_BRANCH` to the branch to test.
4. Run setup cells and confirm Drive is mounted for final archiving only.
5. Set `RUN_DATA_PREP=True` only if Food-101/index files do not already exist in the runtime data directory.
7. Set `RUN_IMAGE_CACHE=True` to create or refresh cached padded-resized images.
8. Set `RUN_TRAINING=True` to run the staged plan.
9. Set `RUN_EVALUATION=True` after training to write metrics/reports into each run.
10. Set `RUN_EXPORT_PT=True` to export `.pt` packages.
11. Set `RUN_ARCHIVE_FINAL_RUNS=True` only for selected final complete runs.
12. Return to `food101_CNN_final_project.ipynb` to load manifests and produce the final comparison.

Resume example:

```python
RESUME_CHECKPOINTS = {
    "resnet50": "/content/food101-runs/resnet50_20260601-2200_v1/checkpoints/resnet50_20260601-2200_v1_best_model.pt",
}
```

If Colab shows only `CalledProcessError` from `scripts/train.py`, rerun the
setup, flags, helpers, and data-source cells before rerunning training. The
notebook helper streams the script output and reports the last command lines on
failure. Uploaded local caches are supported when `cached_index.csv` lives beside
`images/`; stale local absolute paths inside the CSV are relocated automatically
to that uploaded cache directory.

If training fails with `Config file does not exist: configs/<name>.yaml`, check
the `project_root` and `github_branch` printed by the first setup cell. Rerun the
setup cell with `CLONE_FRESH=True` if `/content/food101-cnn` is stale.

Manual Colab command example:

```bash
python scripts/train.py \
  --config configs/resnet50_gpu.yaml \
  --index-csv /content/food101-cnn/data/processed/<cache_hash>/cached_index.csv \
  --epochs 30 \
  --batch-size 32 \
  --device cuda \
  --mixed-precision \
  --num-workers 2 \
  --run-root /content/food101-runs \
  --log-level INFO
```

Manual resume:

```bash
python scripts/train.py \
  --config configs/resnet50_gpu.yaml \
  --index-csv /content/food101-cnn/data/processed/<cache_hash>/cached_index.csv \
  --resume-checkpoint /content/food101-runs/<run_name>/checkpoints/<run_name>_best_model.pt \
  --epochs 10 \
  --device cuda \
  --mixed-precision \
  --run-root /content/food101-runs
```

## End-to-End CLI Workflow

### 1. Download And Index Data

```bash
python scripts/download_data.py \
  --config configs/efficientnet_b0.yaml \
  --index-output data/reports/dataset_index.csv
```

Use `--no-download` if Food-101 is already present under `data/raw/food-101`.

### 2. Validate Images

```bash
python scripts/validate_images.py \
  --config configs/efficientnet_b0.yaml \
  --index-csv data/reports/dataset_index.csv \
  --report-csv data/reports/image_validation_report.csv \
  --corrupted-output data/reports/corrupted_images.txt
```

Validation opens images with PIL, applies EXIF orientation correction, converts to RGB, verifies dimensions/readability, writes a CSV report, and records corrupted paths.

### 3. Cache Resized Images

```bash
python scripts/cache_images.py \
  --config configs/efficientnet_b0.yaml \
  --index-csv data/reports/dataset_index.csv
```

The cache writes EXIF-corrected RGB padded-resized images under `data/processed/<preprocessing_hash>/`, plus `manifest.json` and `cached_index.csv`.

### 4. Train

Small sanity run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/baseline_cnn.yaml \
  --index-csv data/reports/dataset_index.csv \
  --epochs 1 \
  --max-batches 5 \
  --log-level INFO
```

Main EfficientNet-B0 run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/efficientnet_b0.yaml \
  --index-csv data/reports/dataset_index.csv \
  --log-level INFO
```

Training uses Adam, label smoothing, ReduceLROnPlateau, TensorBoard logging, early stopping, and checkpointing by validation top-5 accuracy. Console logs include training setup, epoch start/end, train/validation loss, top-1, top-5, learning rate, checkpoint status, duration, and early-stopping status. Use `--quiet` to suppress per-epoch logs.

### Staged Local Training Schedule

Local runs prioritize feasibility over maximum accuracy. The same CLI supports full training later on a CUDA cloud machine by increasing epochs/batch size in config.

| Stage | Model | Config | Purpose | Local command pattern |
| --- | --- | --- | --- | --- |
| Sanity | `baseline_cnn` | `configs/baseline_cnn.yaml` | Verify data, labels, metrics, logging | `--epochs 1 --max-batches 5` |
| Local-style simple baseline | `baseline_cnn_local` | `configs/baseline_cnn_local.yaml` | Simulate reduced local compute with the lightweight CNN | `--epochs 10 --batch-size 8` |
| Simple full-schedule baseline | `baseline_cnn_simple` | `configs/baseline_cnn_simple.yaml` | Compare the lightweight CNN against the full baseline schedule | `--epochs 30` |
| Baseline full | `baseline_cnn` | `configs/baseline_cnn.yaml` | Pedagogical from-scratch comparison | `--epochs 10` or config default |
| EfficientNet candidate | `efficientnet_b0` | `configs/efficientnet_b0.yaml` | Main local transfer model | config default |
| ResNet comparison | `resnet50` | `configs/resnet50.yaml` | Classical transfer baseline | `--epochs 5` locally, expand in cloud |
| ConvNeXt comparison | `convnext_tiny` | `configs/convnext_tiny.yaml` | Stronger modern CNN candidate | `--epochs 5` locally, expand in cloud |

Recommended remaining local commands:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/baseline_cnn_simple.yaml \
  --index-csv data/reports/dataset_index.csv \
  --epochs 30 \
  --log-level INFO

KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/baseline_cnn_local.yaml \
  --index-csv data/processed/<cache_hash>/cached_index.csv \
  --device auto \
  --batch-size 8 \
  --log-level INFO

KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/baseline_cnn.yaml \
  --index-csv data/reports/dataset_index.csv \
  --epochs 10 \
  --log-level INFO

KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/resnet50.yaml \
  --index-csv data/reports/dataset_index.csv \
  --epochs 5 \
  --log-level INFO

KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py \
  --config configs/convnext_tiny.yaml \
  --index-csv data/reports/dataset_index.csv \
  --epochs 5 \
  --log-level INFO
```

Evaluate each produced run checkpoint before final comparison.

After comparison, write a stable alias for the selected model:

```bash
python scripts/select_final_model.py \
  --run-root outputs/runs \
  --output-dir outputs/final_selected
```

This creates:

- `outputs/final_selected/final_selected_model.pt`
- `outputs/final_selected/manifest.json`

List incomplete run directories before cleanup:

```bash
python scripts/clean_runs.py --run-root outputs/runs
```

A run directory is incomplete when it does not contain `manifest.json`. Deletion
requires explicit confirmation:

```bash
python scripts/clean_runs.py --run-root outputs/runs --delete --yes
```

Update the committed run logbook after final evaluation:

```bash
python scripts/update_run_logbook.py \
  --run-root outputs/runs \
  --output docs/runs_logbook.md
```

The logbook records only small manifest summaries; checkpoints, figures,
TensorBoard logs, and dataset images stay out of Git.

### 5. Monitor TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

### 6. Evaluate

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py \
  --config configs/efficientnet_b0.yaml \
  --checkpoint outputs/runs/<run_name>/checkpoints/<run_name>_best_model.pt \
  --index-csv data/reports/dataset_index.csv \
  --split test
```

Evaluation outputs include:

- `outputs/runs/<run_name>/reports/evaluation_metrics.json`
- `outputs/runs/<run_name>/reports/confusion_matrix.csv`
- `outputs/runs/<run_name>/reports/calibration_report.json`
- `outputs/runs/<run_name>/reports/<run_name>_misclassified.csv`

### 7. Predict

Single image or folder:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/predict.py \
  --config configs/efficientnet_b0.yaml \
  --checkpoint outputs/runs/<run_name>/checkpoints/<run_name>_best_model.pt \
  --input data/processed/test/apple_pie \
  --top-k 5 \
  --threshold 0.40
```

If the maximum softmax probability is below `--threshold`, the result is marked `uncertain`. This is a confidence heuristic only; it does not guarantee out-of-distribution detection.

### 8. Export

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/export_model.py \
  --config configs/efficientnet_b0.yaml \
  --checkpoint outputs/runs/<run_name>/checkpoints/<run_name>_best_model.pt
```

Use `--skip-onnx` when ONNX export dependencies or operators are unavailable.

## Implemented Stages

1. Repository skeleton, environment, README, package markers, pytest setup.
2. YAML config loader, path validation, seed setting, device selection.
3. Food-101 download helper, split-preserving index generation, image validation reports.
4. EXIF/RGB preprocessing, padded resize, ImageNet normalization, train/eval transforms, disk cache.
5. Custom CNN baseline, transfer model factory, classifier replacement, freezing/unfreezing utilities.
6. Training loop, top-1/top-5 metrics, TensorBoard logging, checkpointing, early stopping, scheduler.
7. Macro F1, per-class accuracy, confusion matrix, calibration report, misclassification saving, latency.
8. Prediction CLI, folder inference, top-k probabilities, confidence thresholding, `.pt` and ONNX export.
9. Final executable notebook integration.

## Testing

Run all tests:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest
```

Run focused groups:

```bash
python -m pytest tests/test_config.py tests/test_dataset.py
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_model_forward.py tests/test_training.py
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_evaluation.py tests/test_inference.py
python -m pytest tests/test_notebook.py
```

The tests use synthetic tensors/images and do not require the full Food-101 dataset.

## Output Policy

Do not commit:

- Food-101 images.
- Cached resized images.
- Checkpoints.
- TensorBoard logs.
- Generated reports and figures.
- Local editor settings, personal notes, reference PDFs, and transfer bundles.

Keep final experiment artifacts locally under `outputs/` and dataset-related generated files under `data/reports/`.
Keep only lightweight run summaries such as `docs/runs_logbook.md` in Git.
