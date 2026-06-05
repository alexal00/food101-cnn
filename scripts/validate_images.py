"""Validate Food-101 images and write corruption reports."""

from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.data.dataset import generate_dataset_index
from food101_cnn.data.validation import validate_index, validation_summary
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--index-csv", default="data/reports/dataset_index.csv")
    parser.add_argument(
        "--report-csv",
        default="data/reports/image_validation_report.csv",
    )
    parser.add_argument(
        "--corrupted-output",
        default="data/reports/corrupted_images.txt",
    )
    parser.add_argument(
        "--regenerate-index",
        action="store_true",
        help="Regenerate the dataset index before validating images.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)

    index_csv = resolve_project_path(args.index_csv, project_root)
    report_csv = resolve_project_path(args.report_csv, project_root)
    corrupted_output = resolve_project_path(args.corrupted_output, project_root)

    if args.regenerate_index or not index_csv.is_file():
        data_root = resolve_project_path(config["data"]["root_dir"], project_root)
        generate_dataset_index(
            data_root,
            output_csv=index_csv,
            validation_fraction=float(config["data"]["validation_fraction"]),
            seed=int(config["project"]["seed"]),
        )

    results = validate_index(
        index_csv,
        report_csv=report_csv,
        corrupted_output=corrupted_output,
    )
    summary = validation_summary(results)

    print(f"total={summary['total']}")
    print(f"valid={summary['valid']}")
    print(f"invalid={summary['invalid']}")
    print(f"report_csv={report_csv}")
    print(f"corrupted_output={corrupted_output}")


if __name__ == "__main__":
    main()
