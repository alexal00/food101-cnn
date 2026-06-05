from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from food101_cnn.evaluation.tensorboard_reader import (
    load_tensorboard_scalars,
    save_tensorboard_scalars,
)


def test_load_tensorboard_scalars_from_summary_writer(tmp_path: Path) -> None:
    writer = SummaryWriter(log_dir=str(tmp_path / "run_a" / "tensorboard"))
    writer.add_scalar("train/loss", 1.0, 0)
    writer.add_scalar("validation/top5", 0.75, 1)
    writer.flush()
    writer.close()

    scalars = load_tensorboard_scalars(tmp_path)
    output_csv = save_tensorboard_scalars(scalars, tmp_path / "scalars.csv")

    assert list(scalars.columns) == ["run", "tag", "step", "wall_time", "value"]
    assert set(scalars["tag"]) == {"train/loss", "validation/top5"}
    assert output_csv.is_file()


def test_load_tensorboard_scalars_empty_dir(tmp_path: Path) -> None:
    scalars = load_tensorboard_scalars(tmp_path)

    assert scalars.empty
    assert list(scalars.columns) == ["run", "tag", "step", "wall_time", "value"]
