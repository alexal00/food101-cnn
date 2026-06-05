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
- `configs/baseline_cnn_simple.yaml`: simpler CNN architecture using the same full Colab-style baseline schedule as `baseline_cnn_colab`.
- `configs/baseline_cnn_local.yaml`: the same simpler CNN architecture with conservative Mac/local training limits.
- `configs/resnet50.yaml`: reference transfer-learning backbone.
- `configs/convnext_tiny.yaml`: modern CNN comparison backbone.

Colab configs:

- `configs/baseline_cnn_colab.yaml`
- `configs/efficientnet_b0_colab.yaml`
- `configs/resnet50_colab.yaml`
- `configs/convnext_tiny_colab.yaml`

The Colab configs use Google Drive paths under `/content/drive/MyDrive/food101-cnn/`, CUDA as the preferred device, mixed precision, and moderate batch sizes. The Colab notebook also trains `baseline_cnn_local` and `baseline_cnn_simple` from their main configs so those variants keep distinct run names. CLI overrides can still adjust `--batch-size`, `--device`, `--mixed-precision`, `--no-mixed-precision`, `--gradient-accumulation-steps`, and `--resume-checkpoint`.

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

The notebook is designed to run top-to-bottom after setup. Expensive cells are guarded by flags near the top:

```python
RUN_DATA_PREP = False
RUN_TRAINING = False
RUN_FINAL_EVAL = False
RUN_INFERENCE_EXPORT = False
```

Default execution performs smoke checks only: config loading, transform shapes, model forward pass, synthetic training loop, metrics, calibration, misclassification analysis, and export placeholders.

Recommended notebook usage:

1. Run once with all flags `False` to verify imports and local environment.
2. Run the CLI data preparation commands below to create the dataset index and validation reports.
3. Set `RUN_DATA_PREP = True` only when downloading/indexing/validating Food-101 from inside the notebook is desired.
4. Train models through the CLI for reproducibility and controlled logs.
5. Use notebook sections 10-17 to load latest run manifests and summarize comparison outputs from `outputs/runs/`.
6. Enable `RUN_INFERENCE_EXPORT = True` only after a run checkpoint exists under `outputs/runs/<run_name>/checkpoints/`.

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

Colab cannot see files that exist only on the local machine. To use Colab as a
GPU accelerator, stage a copy of the repository and data into a filesystem Colab
can read:

- Recommended transfer bundle: create a local archive, upload it to Drive, and
  let the notebook unpack it into `/content/food101-cnn`.

```bash
python scripts/create_colab_bundle.py
```

Upload the generated archive:

```text
outputs/colab_transfer/food101_colab_bundle.tar.gz
```

to:

```text
/content/drive/MyDrive/food101-cnn/transfer/food101_colab_bundle.tar.gz
```

Then set this in the first Colab setup cell:

```python
RUN_UNPACK_LOCAL_BUNDLE = True
```

The default bundle includes repository source, configs, scripts, notebooks,
`data/reports`, and `data/processed`. This is enough when the resized cached
images are already available under `data/processed/<hash>/images`. If the
processed cache is not available, create a larger bundle with raw Food-101:

```bash
python scripts/create_colab_bundle.py --include-raw-data
```

Alternative layouts:

- Persistent Drive repo: copy the complete repository to
  `/content/drive/MyDrive/food101-cnn`, including `pyproject.toml`, `configs/`,
  `scripts/`, `src/`, and `notebooks/`.
- Runtime repo: clone/copy the repository to `/content/food101-cnn` each Colab
  session, while keeping data and artifacts in
  `/content/drive/MyDrive/food101-cnn`.
- Custom Drive path: set `REPOSITORY_ROOT_OVERRIDE` in the first Colab notebook
  setup cell to the folder containing `pyproject.toml` and `src/food101_cnn/`.

The notebook adds `src/` to `PYTHONPATH`, so an editable install is optional
when the Colab environment already has the required Python dependencies.

Drive remains the default data source. If local data was staged with the bundle
or manually uploaded with the repository, set this flag in the Colab notebook:

```python
USE_STAGED_LOCAL_DATA = True
```

With that flag enabled, the notebook reads the staged `PROJECT_ROOT/data` for
`reports/dataset_index.csv` and `processed/*/cached_index.csv`. Use
`STAGED_LOCAL_DATA_ROOT_OVERRIDE` only if the uploaded data directory lives
somewhere else. Run artifacts still go to
`/content/drive/MyDrive/food101-cnn/outputs`.

The Colab notebook:

- mounts Google Drive,
- stores persistent data under `/content/drive/MyDrive/food101-cnn/data`,
- optionally unpacks a locally created transfer bundle,
- optionally reads staged local data when `USE_STAGED_LOCAL_DATA=True`,
- stores run artifacts under `/content/drive/MyDrive/food101-cnn/outputs/runs`,
- detects the available CUDA GPU,
- selects moderate batch-size overrides based on GPU memory,
- reuses the latest cached resized-image index when present,
- supports resume checkpoints,
- trains `baseline_cnn_local` for a constrained 10-epoch local-style baseline,
- trains `baseline_cnn_simple` with the same epoch, batch-size, and mixed-precision policy as `baseline_cnn_colab`,
- trains the full comparison set through `scripts/train.py`,
- evaluates each latest run through `scripts/evaluate.py`,
- exports native `.pt` models through `scripts/export_model.py --skip-onnx`.

Colab execution flags:

```python
RUN_DATA_PREP = False
RUN_IMAGE_CACHE = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_EXPORT_PT = False
RUN_UNPACK_LOCAL_BUNDLE = False
USE_STAGED_LOCAL_DATA = False
```

Recommended Colab order:

1. Open `notebooks/food101_colab_training.ipynb` in Colab Pro.
2. Select a GPU runtime.
3. If using the repository already uploaded under
   `/content/drive/MyDrive/food101-cnn`, keep `RUN_UNPACK_LOCAL_BUNDLE=False`.
   If using an archive bundle instead, upload it to Drive and set
   `RUN_UNPACK_LOCAL_BUNDLE=True` in the first setup cell.
4. Run setup cells and confirm Drive is mounted.
5. Keep `USE_STAGED_LOCAL_DATA=False` for Drive data, or set it to `True`
   only after staging/uploading `PROJECT_ROOT/data`.
6. Set `RUN_DATA_PREP=True` only if Food-101/index files do not already exist in the selected data source.
7. Set `RUN_IMAGE_CACHE=True` to create or refresh cached padded-resized images.
8. Set `RUN_TRAINING=True` to run the staged plan.
9. Set `RUN_EVALUATION=True` after training to write metrics/reports into each run.
10. Set `RUN_EXPORT_PT=True` to export `.pt` packages.
11. Return to `food101_CNN_final_project.ipynb` to load manifests and produce the final comparison.

Resume example:

```python
RESUME_CHECKPOINTS = {
    "resnet50": "/content/drive/MyDrive/food101-cnn/outputs/runs/resnet50_20260601-2200_v1/checkpoints/resnet50_20260601-2200_v1_best_model.pt",
}
```

If Colab shows only `CalledProcessError` from `scripts/train.py`, rerun the
setup, flags, helpers, and data-source cells before rerunning training. The
notebook helper streams the script output and reports the last command lines on
failure. Uploaded local caches are supported when `cached_index.csv` lives beside
`images/`; stale local absolute paths inside the CSV are relocated automatically
to that uploaded cache directory.

If training fails with `Config file does not exist: configs/<name>.yaml`, check
the `project_root` printed by the first setup cell. When the repository is
updated directly under `/content/drive/MyDrive/food101-cnn`, keep
`RUN_UNPACK_LOCAL_BUNDLE=False` so the Drive repository is selected. If using an
archive bundle instead, rebuild and re-upload it, then set
`OVERWRITE_UNPACKED_BUNDLE=True` once to replace any stale `/content/food101-cnn`
copy.

Manual Colab command example:

```bash
python scripts/train.py \
  --config configs/resnet50_colab.yaml \
  --index-csv /content/drive/MyDrive/food101-cnn/data/processed/<cache_hash>/cached_index.csv \
  --epochs 30 \
  --batch-size 32 \
  --device cuda \
  --mixed-precision \
  --num-workers 2 \
  --run-root /content/drive/MyDrive/food101-cnn/outputs/runs \
  --log-level INFO
```

Manual resume:

```bash
python scripts/train.py \
  --config configs/resnet50_colab.yaml \
  --index-csv /content/drive/MyDrive/food101-cnn/data/processed/<cache_hash>/cached_index.csv \
  --resume-checkpoint /content/drive/MyDrive/food101-cnn/outputs/runs/<run_name>/checkpoints/<run_name>_best_model.pt \
  --epochs 10 \
  --device cuda \
  --mixed-precision \
  --run-root /content/drive/MyDrive/food101-cnn/outputs/runs
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

Keep final experiment artifacts under `outputs/` and dataset-related generated files under `data/reports/`.
