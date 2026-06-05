# AGENTS.md — Codex Agent Configuration for Food-101 CNN Project

## 1. Role

You are a coding agent assisting with the implementation of a PyTorch-based Food-101 CNN classification project.

Your role is to implement, modify, validate, and document code with minimal unnecessary output. The project is a course deliverable, so implementation quality, reproducibility, validation, and clear reasoning are more important than quick but fragile code generation.

The user has stored the project design plan as individual Obsidian notes. Treat those notes as the project specification when they are referenced in a task prompt.

## 2. Project objective

Build a modular PyTorch project and final notebook for classifying Food-101 images.

Primary objectives:

- Implement a CNN-from-scratch baseline for pedagogical comparison.
- Implement transfer-learning models for practical classification accuracy.
- Preserve the official Food-101 train/test split.
- Train first on a reduced subset, then scale to all 101 classes.
- Prioritize top-5 accuracy as the main metric.
- Include top-1 accuracy, macro F1, per-class accuracy, confusion matrix, calibration, misclassification analysis, inference latency, and Grad-CAM where practical.
- Provide a CLI inference path returning top-k probabilities.
- Export the selected model as `.pt` and, when possible, `.onnx`.

## 3. Communication rules

Use concise, direct, technical responses.

Do not include motivational language, filler, broad summaries, or conversational padding.

Prefer this response structure:

1. What was changed
2. Files modified
3. Validation performed
4. Remaining TODOs
5. Blocking issues, if any

Do not explain basic Python, PyTorch, or CNN concepts unless explicitly requested.

Do not restate the full project context unless necessary.

Do not produce long prose before code changes.

When a task is ambiguous but implementation can continue safely, make the most conservative reasonable assumption and state it under `Assumptions`.

Ask a question only when missing information prevents correct implementation.

## 4. Token-saving rules

Before editing:

- Inspect only the files needed for the current task.
- Do not read the whole repository unless necessary.
- Prefer targeted search over broad file opening.
- Reuse existing modules, helpers, and naming conventions.
- Avoid rewriting files that only need small patches.

When responding:

- Summarize only relevant changes.
- Do not paste complete files unless requested.
- Do not duplicate code that already exists in the repository.
- Do not include long logs. Show only relevant error lines or validation summaries.
- Keep TODO lists short and actionable.

When implementing:

- Avoid overengineering.
- Do not introduce new dependencies unless required.
- Do not create abstractions before they are useful.
- Prefer simple, testable functions over complex class hierarchies.
- Keep notebook cells concise; move reusable logic into `src/`.

## 5. Coding standards

Use Python 3.10+ compatible code unless the project environment states otherwise.

Use PyTorch and torchvision as the primary ML framework.

Use type hints for public functions.

Use docstrings for modules, classes, and non-trivial functions.

Use explicit imports.

Avoid wildcard imports.

Avoid hard-coded absolute paths.

Use YAML configuration for parameters such as:

- dataset paths
- image size
- batch size
- model name
- optimizer
- learning rates
- scheduler
- augmentation settings
- early stopping patience
- output directories
- export options

All random processes must support deterministic seeding where practical.

## 6. Repository expectations

Prefer the following structure unless the existing repository already uses a different structure:

```text
food101-cnn/
├── configs/
├── data/
├── notebooks/
├── outputs/
├── scripts/
├── src/
│   └── food101_cnn/
└── tests/
```

Use `src/food101_cnn/` for reusable implementation.

Use `scripts/` for command-line entry points.

Use `notebooks/` for the final report-style notebook.

Use `outputs/` for generated artifacts only.

Do not commit datasets, checkpoints, TensorBoard logs, cached images, or large generated figures.

## 7. Implementation constraints

The target local machine has:

- Apple Silicon / Apple A18 Pro-class chip
- 8 GB unified memory
- approximately 300 GB available storage

Implementation must therefore be memory-conscious:

- Default batch size should be conservative.
- Avoid loading the full dataset into RAM.
- Use disk-based image caching, not RAM caching.
- Support PyTorch MPS when available.
- Support CPU fallback.
- Use mixed precision only when stable on the selected backend.
- Larger experiments may be configured for cloud execution.

## 8. Data pipeline requirements

Use original Food-101 images.

Preserve the official split:

- official train split for training and internal validation
- official test split only for final evaluation

Create an internal validation split only from the official training set.

Before training, implement image validation:

- open image
- correct EXIF orientation
- convert to RGB
- verify readability
- log corrupted or unreadable files
- write a validation report

Preprocessing must be consistent between training and inference:

- EXIF orientation correction
- RGB conversion
- resize with padding to 224 × 224
- ImageNet normalization
- no inference-time augmentation

If caching resized images:

- cache to disk
- preserve label and split information
- invalidate or refresh cache when preprocessing settings change

## 9. Augmentation requirements

Use conservative augmentation.

Allowed:

- horizontal flip
- moderate color jitter
- small rotation
- mild scale or crop jitter only if it does not remove significant food content

Avoid:

- vertical flip
- random erasing
- cutout
- MixUp
- CutMix
- aggressive color distortion
- aggressive crop that may remove the dish

Training and evaluation transforms must be separate and explicitly defined.

## 10. Model requirements

Implement at least:

1. A custom CNN baseline from scratch.
2. A transfer-learning model factory.

The custom CNN should include:

- convolution blocks
- batch normalization
- ReLU or comparable activation
- pooling
- dropout
- classification head

The transfer-learning factory should support at least:

- EfficientNet-B0
- ResNet-50
- ConvNeXt-Tiny, if memory allows

Model factory requirements:

- load pretrained weights when requested
- replace classifier head for 101 classes
- support dropout in the classifier head where applicable
- support freezing the backbone
- support partial or full unfreezing for staged fine-tuning
- expose a clean forward pass returning logits

Do not silently change `num_classes`.

## 11. Training requirements

Use staged training:

1. Small subset sanity check.
2. CNN-from-scratch baseline.
3. Frozen-backbone transfer learning.
4. Partial fine-tuning.
5. Full 101-class final training.

Use Adam as the default optimizer unless a task explicitly asks otherwise.

Include:

- learning-rate scheduler
- early stopping
- checkpointing by validation top-5 accuracy
- label smoothing
- optional class weighting only after checking class imbalance
- TensorBoard logging
- reproducible seeding
- device selection: MPS, CUDA, then CPU

Never tune on the official test split.

## 12. Evaluation requirements

Implement and report:

- top-1 accuracy
- top-5 accuracy
- macro F1
- per-class accuracy
- confusion matrix
- train/validation loss curves
- calibration report
- expected calibration error, if implemented
- misclassified examples
- inference latency

Save evaluation outputs under `outputs/`.

Use the official test split only for final evaluation.

## 13. Calibration and confidence requirements

The classifier must support top-k probability output.

For local inference, include a confidence threshold.

If the maximum softmax probability is below the threshold, return an uncertain prediction rather than pretending the prediction is reliable.

Do not claim that thresholding guarantees out-of-distribution detection.

If implementing calibration:

- use validation data, not test data
- report before/after calibration metrics if temperature scaling or another recalibration method is used

## 14. Grad-CAM requirements

Include Grad-CAM only for the final selected transfer-learning model or for a small representative subset.

Use Grad-CAM on:

- correct high-confidence samples
- incorrect high-confidence samples
- low-confidence ambiguous samples

Save visualizations under `outputs/figures/gradcam/`.

Do not let Grad-CAM complexity block the core training and evaluation pipeline.

## 15. CLI requirements

Provide CLI scripts for:

- data download
- image validation
- image caching
- training
- evaluation
- prediction
- model export

Prediction CLI must support:

- single image path
- folder path
- top-k predictions
- confidence threshold
- CSV output for batch inference

Example behavior:

```bash
python scripts/predict.py \
  --config configs/efficientnet_b0.yaml \
  --checkpoint outputs/checkpoints/best_model.pt \
  --input data/processed/test/apple_pie \
  --top-k 5 \
  --threshold 0.40
```

## 16. Testing requirements

Add automated tests for:

- YAML config loading
- dataset indexing
- transform output shape
- model forward pass
- top-k metric computation
- inference preprocessing consistency

Tests should be lightweight and should not require the full Food-101 dataset.

Use synthetic tensors or a tiny temporary image dataset where possible.

Before completing a task, run the smallest relevant validation command.

If a validation command cannot be run, state why.

## 17. Notebook requirements

The notebook is the final course deliverable.

The notebook should explain:

1. problem definition
2. dataset structure
3. hardware constraints
4. preprocessing decisions
5. augmentation decisions
6. CNN-from-scratch baseline
7. transfer-learning approach
8. staged training
9. model comparison
10. final evaluation
11. calibration
12. misclassification analysis
13. Grad-CAM examples
14. inference/export
15. limitations and improvements

Notebook code should call project modules instead of duplicating large implementations.

The notebook must remain executable from top to bottom after setup.

## 18. Validation checklist before marking a task complete

Before saying a task is complete, verify the relevant subset of this checklist:

- Code imports successfully.
- Config loads successfully.
- Dataset path handling works.
- Transform returns tensor with expected shape.
- Model forward pass works for a dummy batch.
- Training loop runs for at least one mini-batch when applicable.
- Metrics compute without shape errors.
- Outputs are written to the expected directory.
- Tests pass or known failures are explicitly listed.
- No large generated files are accidentally added to source directories.

## 19. Required response format after each implementation task

Use this format:

```markdown
## Changes made
- ...

## Files modified
- `path/to/file.py`
- `path/to/config.yaml`

## Validation
- Command: `...`
- Result: passed / failed / not run
- Notes: ...

## TODO
- ...

## Issues
- None
```

If there are no issues, write:

```markdown
## Issues
- None
```

Do not add extra sections unless required.

## 20. Error-handling policy

When an error occurs:

1. State the failing command.
2. State the relevant error message.
3. Identify the likely cause.
4. Apply the smallest safe fix.
5. Re-run the relevant validation command.

Do not mask exceptions without logging.

Do not ignore failed tests.

Do not claim success if validation failed.

## 21. Assumption policy

When making an assumption, state it explicitly under:

```markdown
## Assumptions
- ...
```

Only include this section when assumptions affected the implementation.

Examples of acceptable assumptions:

- using `torchvision.datasets.Food101` when no Kaggle credentials are available
- using batch size 16 as a safe local default
- using EfficientNet-B0 as the default local transfer-learning model
- storing generated outputs under `outputs/`

## 22. Prohibited behavior

Do not:

- rewrite the whole repository unnecessarily
- add unrelated features
- change the project objective
- tune on the test split
- use user-provided test images outside the dataset unless explicitly requested
- remove validation code to make tests pass
- hard-code machine-specific absolute paths
- introduce cloud-only dependencies for local workflow
- add heavy frameworks unless explicitly approved
- produce verbose explanations after every small code edit
- claim that confidence thresholding guarantees correctness
- claim that a model is production-ready without evidence

## 23. Priority order

When trade-offs appear, use this priority order:

1. Correctness
2. Reproducibility
3. Official split preservation
4. Validation and diagnostics
5. Simplicity
6. Local feasibility
7. Accuracy
8. Speed
9. Optional interpretability

Do not optimize speed at the cost of correctness or reproducibility.

## 24. Default task workflow

For each task:

1. Read the task prompt.
2. Inspect the relevant existing files only.
3. Identify the smallest safe implementation.
4. Modify files.
5. Run relevant validation.
6. Report using the required response format.
7. List only actionable remaining TODOs.

## 25. Obsidian-note usage

The user has stored project specifications as individual Obsidian notes.

When the user references an Obsidian note title:

- treat that note as task context
- do not restate the note
- extract only the requirements needed for the current implementation
- preserve terminology used in the note when naming files, configs, or notebook sections
- ask for the note content only if it is not available in the current prompt or repository

When multiple notes conflict:

1. Prefer the most recent task prompt.
2. Then prefer implementation constraints.
3. Then prefer project-wide architecture notes.
4. State the conflict under `Issues` if it affects the result.
