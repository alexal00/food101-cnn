"""Create a local repo/data archive for Colab GPU training."""

from argparse import ArgumentParser

from food101_cnn.utils.colab_bundle import create_colab_bundle
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/colab_transfer/food101_colab_bundle.tar.gz",
        help="Destination .tar.gz archive to upload to Google Drive.",
    )
    parser.add_argument(
        "--include-raw-data",
        action="store_true",
        help="Include data/raw in addition to processed cache and reports.",
    )
    parser.add_argument(
        "--no-processed-data",
        action="store_true",
        help="Do not include data/processed.",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not include data/reports.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    output_path = resolve_project_path(args.output, project_root)

    summary = create_colab_bundle(
        project_root,
        output_path,
        include_processed_data=not args.no_processed_data,
        include_reports=not args.no_reports,
        include_raw_data=args.include_raw_data,
    )

    print(f"bundle={summary.output_path}")
    print(f"manifest={summary.manifest_path}")
    print(f"total_files={summary.total_files}")
    print(f"total_bytes={summary.total_bytes}")
    print("included_roots=" + ",".join(summary.included_roots))


if __name__ == "__main__":
    main()
