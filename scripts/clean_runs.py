"""List or delete incomplete run directories under outputs/runs."""

from argparse import ArgumentParser
import shutil

from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.runs import list_incomplete_run_dirs


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete listed incomplete run directories.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --delete to confirm destructive cleanup.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    run_root = resolve_project_path(args.run_root, project_root)
    incomplete = list_incomplete_run_dirs(project_root, run_root=run_root)

    print(f"run_root={run_root}")
    print(f"incomplete_count={len(incomplete)}")
    for item in incomplete:
        print(f"incomplete_run={item.run_dir} reason={item.reason}")

    if args.delete and not args.yes:
        raise SystemExit("Refusing to delete without --yes.")

    if args.delete and args.yes:
        for item in incomplete:
            shutil.rmtree(item.run_dir)
            print(f"deleted_run={item.run_dir}")


if __name__ == "__main__":
    main()
