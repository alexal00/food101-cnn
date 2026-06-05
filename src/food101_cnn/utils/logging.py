"""Project logging helpers."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise console logging for CLI scripts."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
