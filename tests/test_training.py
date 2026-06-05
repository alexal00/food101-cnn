from pathlib import Path
import logging

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from food101_cnn.training.callbacks import CheckpointManager, EarlyStopping
from food101_cnn.training.loops import train_one_epoch, validate_one_epoch
from food101_cnn.training.losses import build_classification_loss
from food101_cnn.training.schedulers import build_scheduler
from food101_cnn.training.train import build_optimizer, fit


def test_train_and_validate_one_epoch_on_synthetic_data() -> None:
    model = _tiny_model(num_classes=3)
    loader = _tiny_loader(num_classes=3)
    criterion = build_classification_loss(label_smoothing=0.1)
    optimizer = build_optimizer(model, lr=0.01)

    train_metrics = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        "cpu",
        max_batches=1,
    )
    val_metrics = validate_one_epoch(
        model,
        loader,
        criterion,
        "cpu",
        max_batches=1,
    )

    assert train_metrics.samples == 4
    assert val_metrics.samples == 4
    assert 0 <= train_metrics.top1 <= 1
    assert 0 <= train_metrics.top5 <= 1
    assert train_metrics.loss > 0


def test_checkpoint_manager_saves_only_improvements(tmp_path: Path) -> None:
    model = _tiny_model(num_classes=3)
    optimizer = build_optimizer(model, lr=0.01)
    manager = CheckpointManager(tmp_path, metric_name="val_top5", mode="max")

    assert manager.step(
        model=model,
        optimizer=optimizer,
        epoch=0,
        metric_value=0.5,
    )
    assert manager.checkpoint_path.is_file()

    checkpoint = torch.load(manager.checkpoint_path, map_location="cpu")
    assert checkpoint["metric_name"] == "val_top5"
    assert checkpoint["metric_value"] == 0.5

    assert not manager.step(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metric_value=0.4,
    )


def test_early_stopping_uses_patience() -> None:
    stopper = EarlyStopping(patience=1, mode="max")

    assert not stopper.step(0.5)
    assert not stopper.step(0.4)
    assert stopper.step(0.3)


def test_scheduler_reduces_learning_rate_on_plateau() -> None:
    model = _tiny_model(num_classes=3)
    optimizer = build_optimizer(model, lr=0.01)
    scheduler = build_scheduler(
        optimizer,
        "reduce_on_plateau",
        mode="max",
        patience=0,
        factor=0.5,
    )

    assert scheduler is not None
    scheduler.step(0.5)
    scheduler.step(0.4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_fit_runs_sanity_training_stage(tmp_path: Path) -> None:
    model = _tiny_model(num_classes=3)
    loader = _tiny_loader(num_classes=3)
    config = {
        "training": {
            "label_smoothing": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "early_stopping_patience": 2,
            "mixed_precision": False,
        }
    }

    history = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        device="cpu",
        epochs=1,
        lr=0.01,
        checkpoint_dir=tmp_path / "checkpoints",
        tensorboard_dir=tmp_path / "tensorboard",
        max_batches=1,
    )

    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "best_model.pt").is_file()
    assert list((tmp_path / "tensorboard").glob("events.out.tfevents.*"))


def test_fit_logs_epoch_progress(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    model = _tiny_model(num_classes=3)
    loader = _tiny_loader(num_classes=3)
    config = {
        "training": {
            "label_smoothing": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "early_stopping_patience": 2,
            "mixed_precision": False,
        }
    }

    with caplog.at_level(logging.INFO, logger="food101_cnn.training.train"):
        fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
            epochs=1,
            lr=0.01,
            checkpoint_dir=tmp_path / "checkpoints",
            tensorboard_dir=None,
            max_batches=1,
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "training_start" in log_text
    assert "epoch=1/1 status=finished" in log_text
    assert "train_loss=" in log_text
    assert "val_top5=" in log_text


def test_fit_resumes_from_checkpoint(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = {
        "training": {
            "label_smoothing": 0.0,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "early_stopping_patience": 2,
            "mixed_precision": False,
        }
    }
    loader = _tiny_loader(num_classes=3)

    fit(
        model=_tiny_model(num_classes=3),
        train_loader=loader,
        val_loader=loader,
        config=config,
        device="cpu",
        epochs=1,
        lr=0.01,
        checkpoint_dir=tmp_path / "checkpoints",
        max_batches=1,
        log_progress=False,
    )

    checkpoint_path = tmp_path / "checkpoints" / "best_model.pt"
    with caplog.at_level(logging.INFO, logger="food101_cnn.training.train"):
        history = fit(
            model=_tiny_model(num_classes=3),
            train_loader=loader,
            val_loader=loader,
            config=config,
            device="cpu",
            epochs=1,
            lr=0.01,
            checkpoint_dir=tmp_path / "checkpoints",
            max_batches=1,
            resume_checkpoint=checkpoint_path,
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert len(history) == 1
    assert "start_epoch=1" in log_text


def _tiny_model(num_classes: int) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


def _tiny_loader(num_classes: int) -> DataLoader:
    images = torch.randn(8, 3, 8, 8)
    labels = torch.arange(8) % num_classes
    return DataLoader(TensorDataset(images, labels), batch_size=4)
