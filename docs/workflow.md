# Food-101 Workflow Contract

This project has one source of truth for code and three separate places for
execution artifacts.

## Source Of Truth

- The `main` branch on GitHub is the canonical project source.
- Cleanup and refactor work happens on short-lived branches, then merges back
  into `main` after review.
- Google Drive must not contain a working repository copy. A stale Drive repo can
  silently diverge from GitHub and make results hard to reproduce.

## Storage Boundaries

- Local machine:
  - Food-101 raw data.
  - Processed image caches.
  - Local development outputs and exploratory runs.
- Colab runtime:
  - Temporary cloned repository.
  - Temporary downloaded data/cache for cloud training.
  - Temporary run outputs before a selected run is archived.
- Google Drive:
  - Final selected runs only.
  - Final checkpoints, exported models, manifests, and summary reports.
  - No repository clone, broad dataset copy, partial runs, or routine logs.

## Primary Execution Path

The primary workflow is local CLI execution from the repository root:

```bash
python scripts/download_data.py --config configs/efficientnet_b0.yaml
python scripts/train.py --config configs/efficientnet_b0.yaml
python scripts/evaluate.py --config configs/efficientnet_b0.yaml --checkpoint outputs/runs/<run>/checkpoints/<checkpoint>.pt
```

Configuration files use project-relative paths. The `*_gpu.yaml` profiles keep
the stronger CUDA-oriented batch sizes, epoch counts, and mixed-precision
settings, but they do not point at Google Drive.

For staged cloud/debug sessions, use the plan runner instead of rebuilding
script commands in notebooks:

```bash
python scripts/run_experiment_plan.py --plan gpu-default --train --evaluate --export-pt
```

The plan runner delegates to the stable CLI scripts: `download_data.py`,
`cache_images.py`, `train.py`, `evaluate.py`, and `export_model.py`.

Notebook usage is split by purpose:

- `notebooks/food101_CNN_final_project.ipynb` is the final report notebook.
  It loads existing local artifacts and keeps data preparation, training,
  evaluation, post-processing, and export flags disabled by default.
- `notebooks/food101_colab_training.ipynb` is a training/debug notebook for
  cloud GPU sessions and metric generation.

## Colab Contract

Colab uses GitHub for code:

1. Mount Drive only so selected final artifacts can be archived.
2. Clone the configured GitHub branch into `/content/food101-cnn`.
3. Install the package in editable mode.
4. Download/cache Food-101 into the temporary Colab runtime, not Drive.
5. Train and evaluate through repository scripts.
6. Copy only selected complete run directories to Drive.

Incomplete run directories are disposable. A run is considered complete only
when its directory contains `manifest.json`.

Run hygiene is handled through a dry-run-first CLI:

```bash
python scripts/clean_runs.py --run-root outputs/runs
```

Deletion requires explicit confirmation:

```bash
python scripts/clean_runs.py --run-root outputs/runs --delete --yes
```

## Run Logbook

The repository should keep a small committed run logbook for finished relevant
runs. The logbook records model name, config, run date, metrics, artifact
location, and notes. It does not store checkpoints or generated figures.

Regenerate it from completed manifests after final evaluation:

```bash
python scripts/update_run_logbook.py \
  --run-root outputs/runs \
  --output docs/runs_logbook.md
```

By default this records only the latest completed run per model. Use
`--include-all` when the course report needs the full completed-run history.

## GitHub Publishing

Use a clean snapshot branch for final publishing if the old branch history still
contains data archives, checkpoints, exported models, or run products. The
latest tree can be small while parent commits still make a normal push too
large. See `docs/publishing.md` for the branch-size check.
