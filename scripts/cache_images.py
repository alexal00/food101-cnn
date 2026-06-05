"""Generate a disk cache of resized Food-101 images."""

from argparse import ArgumentParser

from food101_cnn.config import load_config
from food101_cnn.data.cache import cache_resized_images
from food101_cnn.data.dataset import generate_dataset_index
from food101_cnn.utils.paths import find_project_root, resolve_project_path


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/efficientnet_b0.yaml")
    parser.add_argument("--index-csv", default="data/reports/dataset_index.csv")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--regenerate-index",
        action="store_true",
        help="Regenerate the dataset index before caching images.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    config = load_config(args.config, project_root=project_root, create_missing_dirs=True)

    index_csv = resolve_project_path(args.index_csv, project_root)
    cache_root = resolve_project_path(
        args.cache_root or config["data"]["processed_dir"],
        project_root,
    )

    if args.regenerate_index or not index_csv.is_file():
        data_root = resolve_project_path(config["data"]["root_dir"], project_root)
        generate_dataset_index(
            data_root,
            output_csv=index_csv,
            validation_fraction=float(config["data"]["validation_fraction"]),
            seed=int(config["project"]["seed"]),
        )

    summary = cache_resized_images(
        index_csv,
        cache_root,
        image_size=args.image_size or int(config["data"]["image_size"]),
        refresh=args.refresh,
    )

    print(f"total={summary.total}")
    print(f"cached={summary.cached}")
    print(f"skipped={summary.skipped}")
    print(f"cache_dir={summary.cache_dir}")
    print(f"manifest={summary.manifest_path}")
    print(f"cached_index={summary.cached_index_path}")


if __name__ == "__main__":
    main()
