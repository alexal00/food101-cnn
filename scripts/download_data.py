"""Download Food-101 and generate the local dataset index."""

from argparse import ArgumentParser
from pathlib import Path

from food101_cnn.config import load_config
from food101_cnn.data.dataset import generate_dataset_index
from food101_cnn.data.download import download_food101, verify_food101_structure
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--index-output", default="data/reports/dataset_index.csv")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)

    data_root = resolve_project_path(config["data"]["root_dir"], project_root)
    index_output = resolve_project_path(args.index_output, project_root)

    download_food101(data_root, download=not args.no_download)
    summary = verify_food101_structure(data_root)
    records = generate_dataset_index(
        data_root,
        output_csv=index_output,
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["project"]["seed"]),
    )

    print(f"dataset_dir={summary['dataset_dir']}")
    print(f"class_count={summary['class_count']}")
    print(f"train_count={summary['train_count']}")
    print(f"test_count={summary['test_count']}")
    print(f"index_rows={len(records)}")
    print(f"index_csv={Path(index_output)}")


if __name__ == "__main__":
    main()
