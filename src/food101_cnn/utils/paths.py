"""Path helpers for repository-relative configuration values."""

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest project root containing ``pyproject.toml``."""
    start_path = Path.cwd() if start is None else Path(start)
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    return current


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve an absolute or project-relative path."""
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path(project_root).resolve() / path_obj).resolve()
