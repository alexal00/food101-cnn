import json
from pathlib import Path


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def test_final_notebook_is_valid_json_and_has_required_sections() -> None:
    notebook_path = Path("notebooks/food101_CNN_final_project.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    markdown = "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    required_headings = [
        "## 1. Introduction",
        "## 2. Dataset Description",
        "## 3. Hardware and Implementation Constraints",
        "## 4. Project Methodology",
        "## 5. Data Acquisition and Validation",
        "## 6. Preprocessing and Augmentation",
        "## 7. Baseline CNN From Scratch",
        "## 8. Transfer Learning Design",
        "## 9. Training Strategy",
        "## 10. Experiment 1: Small Subset Sanity Check",
        "## 11. Experiment 2: Baseline CNN",
        "## 12. Experiment 3: EfficientNet-B0 Transfer Learning",
        "## 13. Experiment 4: Backbone Comparison",
        "## 14. Final 101-Class Model",
        "## 15. Evaluation",
        "## 16. Calibration Analysis",
        "## 17. Misclassification Analysis",
        "## 18. Post-processing and Debugging Analysis",
        "## 19. Inference and Export",
        "## 20. Conclusions and Recommended Improvements",
    ]

    for heading in required_headings:
        assert heading in markdown

    postprocess_sections = [
        "Model Comparison Overview",
        "Baseline CNN Results",
        "Simpler CNN Full-Schedule Results",
        "EfficientNet-B0 Results",
        "ResNet-50 Results",
        "ConvNeXt-Tiny Results",
        "Local CNN Baseline Results",
        "Final Model Selection",
        "Baseline CNN Qualitative Top-5 Predictions",
        "Baseline CNN Feature-Space Nearest Neighbors",
        "Data Augmentation and Overfitting",
        "Pooling Placement in the Baseline CNN",
        "Training Curves from TensorBoard Logs",
        "Overfitting and Underfitting Diagnostics",
        "Class-Level Performance Analysis",
        "Best and Worst Performing Food Classes",
        "Confusion Matrix Analysis",
        "Most Common Misclassifications",
        "Misclassified Image Galleries",
        "Calibration and Confidence Analysis",
        "Grad-CAM Interpretability",
        "Model Improvement Recommendations",
    ]
    for heading in postprocess_sections:
        assert heading in markdown

    assert "final-report notebook expects to run from a local or cloned repository checkout" in markdown
    assert "NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf" not in markdown

    code = "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "RUN_DATA_PREP = False" in code
    assert "RUN_TRAINING = False" in code
    assert "RUN_FINAL_EVAL = False" in code
    assert "RUN_POSTPROCESSING = False" in code
    assert "RUN_INFERENCE_EXPORT = False" in code
    assert "RUN_UNPACK_LOCAL_BUNDLE" not in code
    assert "REPOSITORY_ROOT_OVERRIDE" not in code
    assert "COLAB_BUNDLE_PATH" not in code
    assert "google.colab" not in code
    assert "tarfile" not in code
    assert "DRIVE_PROJECT_ROOT" not in code

def test_colab_training_notebook_is_valid_json() -> None:
    notebook_path = Path("notebooks/food101_colab_training.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    markdown = "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "Food-101 Colab GPU Training Workflow" in markdown
    assert "Run Manifest Snapshot" in markdown

    code = "\n".join(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "GITHUB_REPO_URL" in code
    assert "GITHUB_BRANCH" in code
    assert "git" in code
    assert "clone" in code
    assert "PYTHONPATH" in code
    assert "RUN_ARCHIVE_FINAL_RUNS" in code
    assert "DRIVE_FINAL_RUN_ROOT" in code
    assert "PROJECT_ROOT / \"data\"" in code
    assert "RUN_UNPACK_LOCAL_BUNDLE" not in code
    assert "REPOSITORY_ROOT_OVERRIDE" not in code
    assert "USE_STAGED_LOCAL_DATA" not in code
    assert "scripts/run_experiment_plan.py" in code
    assert "build_plan_command" in code
    assert "TRAINING_PLAN" not in code
    assert "train_command" not in code
    assert "evaluate_latest_command" not in code
    assert "Last command output" in code
    assert "import food101_cnn" in code
    assert "PLAN_NAME" in code
    assert "PLAN_MODELS" in code
