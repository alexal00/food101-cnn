"""Write the committed run logbook from completed run manifests."""

from argparse import ArgumentParser

from food101_cnn.utils.paths import find_project_root, resolve_project_path
from food101_cnn.utils.run_logbook import (
    build_run_logbook_entries,
    render_run_logbook_markdown,
)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/runs")
    parser.add_argument("--output", default="docs/runs_logbook.md")
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include every completed run instead of only the latest run per model.",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()
    project_root = find_project_root()
    run_root = resolve_project_path(args.run_root, project_root)
    output_path = resolve_project_path(args.output, project_root)

    entries = build_run_logbook_entries(
        project_root,
        run_root=run_root,
        latest_only=not args.include_all,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_run_logbook_markdown(entries), encoding="utf-8")

    print(f"run_root={run_root}")
    print(f"logbook_entries={len(entries)}")
    print(f"logbook_path={output_path}")


if __name__ == "__main__":
    main()
