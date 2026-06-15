# Installation and Usage

This manual describes the recommended local workflow for the final Food-101 CNN
deliverable. It complements the project overview in `README.md`.

## 1. Clone the Repository

The repository uses Git LFS for large processed images and trained model files.
Install Git LFS before cloning or before pulling the final artifacts.

```bash
git lfs install
git clone https://github.com/alexal00/food101-cnn.git
cd food101-cnn
git lfs pull
```

After `git lfs pull`, the processed image cache and model `.pt` files should be
real binary files, not small text pointer files.

## 2. Create the Environment with Conda

The project environment is defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate food101-cnn
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate food101-cnn
```

## 3. Create the Environment with pip

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra includes the notebook tooling used for validation:
`ipykernel`, `jupyterlab`, `nbconvert`, `nbformat`, and `pytest`.

## 4. Dataset Placement

The configured dataset root is `data/raw`. If you provide the original Food-101
dataset manually, place it like this:

```text
data/raw/food-101/
├── images/
└── meta/
    ├── train.txt
    └── test.txt
```

For the final deliverable snapshot, the processed cache is versioned under:

```text
data/processed/f8e106c00724/
├── cached_index.csv
├── manifest.json
└── images/
```

That cache is enough for the final report notebook and local artifact review
after Git LFS files have been pulled. If you want to regenerate the raw index
from a manually placed Food-101 folder, run:

```bash
python scripts/download_data.py \
  --config configs/efficientnet_b0.yaml \
  --no-download \
  --index-output data/reports/dataset_index.csv
```

Then refresh the processed cache if needed:

```bash
python scripts/cache_images.py \
  --config configs/efficientnet_b0.yaml \
  --index-csv data/reports/dataset_index.csv
```

## 5. Run the Final Notebook

Open the final notebook locally:

```bash
jupyter lab notebooks/food101_CNN_final_project.ipynb
```

The default notebook mode is report mode. These flags are intentionally disabled
near the top of the notebook:

```python
RUN_DATA_PREP = False
RUN_TRAINING = False
RUN_FINAL_EVAL = False
RUN_POSTPROCESSING = False
RUN_INFERENCE_EXPORT = False
```

With the committed artifacts present, run all notebook cells from top to bottom.
The notebook loads the saved runs, reports, figures, checkpoints, and cached
feature files instead of retraining the full model suite.

To execute and overwrite the notebook outputs from the command line:

```bash
KMP_DUPLICATE_LIB_OK=TRUE jupyter nbconvert \
  --to notebook \
  --execute notebooks/food101_CNN_final_project.ipynb \
  --output food101_CNN_final_project.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=900
```

On macOS Conda environments, `KMP_DUPLICATE_LIB_OK=TRUE` avoids a known duplicate
OpenMP runtime abort during Torch imports.

## 6. Final Model and Artifacts

The stable selected-model alias is:

```text
outputs/final_selected/final_selected_model.pt
outputs/final_selected/manifest.json
```

The source run selected by top-5 accuracy, then macro F1, is recorded in the
final selected manifest. Completed run artifacts live under:

```text
outputs/runs/<run_name>/
├── checkpoints/
├── exports/
├── figures/
├── reports/
└── manifest.json
```

Global figures and report tables used by the notebook live under:

```text
outputs/figures/
outputs/reports/
```

## 7. Validate the Repository

Run the lightweight test suite:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest
```

The tests use synthetic data for most checks and do not retrain the final
models.

## 8. Optional Regeneration Workflow

Training from scratch is expensive. Use the default final notebook mode for
normal review. To regenerate artifacts manually, use the scripts in this order:

```bash
python scripts/download_data.py --config configs/efficientnet_b0.yaml --no-download
python scripts/cache_images.py --config configs/efficientnet_b0.yaml
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --config configs/efficientnet_b0.yaml --index-csv data/reports/dataset_index.csv
KMP_DUPLICATE_LIB_OK=TRUE python scripts/evaluate.py --config configs/efficientnet_b0.yaml --checkpoint outputs/runs/<run_name>/checkpoints/<run_name>_best_model.pt --index-csv data/reports/dataset_index.csv --split test
python scripts/postprocess_runs.py --config configs/efficientnet_b0.yaml --log-dir outputs/runs --output-dir outputs
python scripts/select_final_model.py --run-root outputs/runs --output-dir outputs/final_selected
```

Replace `<run_name>` with the generated run directory name.
