"""Create a portable repository/data bundle for Colab training."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from food101_cnn.utils.runs import LOCAL_TIMEZONE


DEFAULT_SOURCE_PATHS = (
    "pyproject.toml",
    "README.md",
    "environment.yml",
    "configs",
    "scripts",
    "src",
    "notebooks",
)


@dataclass(frozen=True)
class ColabBundleSummary:
    """Metadata for a created Colab transfer bundle."""

    output_path: Path
    manifest_path: Path
    total_files: int
    total_bytes: int
    included_roots: list[str]


def create_colab_bundle(
    project_root: str | Path,
    output_path: str | Path,
    *,
    include_processed_data: bool = True,
    include_reports: bool = True,
    include_raw_data: bool = False,
    archive_root_name: str = "food101-cnn",
) -> ColabBundleSummary:
    """Create a compressed archive that Colab can unpack and train from."""
    root = Path(project_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = _bundle_entries(
        root,
        include_processed_data=include_processed_data,
        include_reports=include_reports,
        include_raw_data=include_raw_data,
    )
    total_files = 0
    total_bytes = 0

    with tarfile.open(output, "w:gz") as archive:
        for relative_path in entries:
            source = root / relative_path
            if not source.exists():
                continue
            arcname = Path(archive_root_name) / relative_path
            archive.add(source, arcname=arcname, recursive=True)
            files = _iter_files(source)
            for file_path in files:
                total_files += 1
                total_bytes += file_path.stat().st_size

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    included_roots = [str(path) for path in entries if (root / path).exists()]
    manifest = {
        "created_at": datetime.now(ZoneInfo(LOCAL_TIMEZONE)).isoformat(timespec="seconds"),
        "archive_root_name": archive_root_name,
        "included_roots": included_roots,
        "include_processed_data": include_processed_data,
        "include_reports": include_reports,
        "include_raw_data": include_raw_data,
        "output_path": str(output),
        "total_bytes": total_bytes,
        "total_files": total_files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return ColabBundleSummary(
        output_path=output,
        manifest_path=manifest_path,
        total_files=total_files,
        total_bytes=total_bytes,
        included_roots=included_roots,
    )


def _bundle_entries(
    root: Path,
    *,
    include_processed_data: bool,
    include_reports: bool,
    include_raw_data: bool,
) -> list[Path]:
    entries = [Path(path) for path in DEFAULT_SOURCE_PATHS]
    data_entries: list[Path] = []
    if include_reports:
        data_entries.append(Path("data/reports"))
    if include_processed_data:
        data_entries.append(Path("data/processed"))
    if include_raw_data:
        data_entries.append(Path("data/raw"))

    return [path for path in [*entries, *data_entries] if (root / path).exists()]


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in path.rglob("*"):
        if item.is_file():
            yield item
