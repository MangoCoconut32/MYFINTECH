"""Central logging configuration for the MYFINTECH pipeline.

This module must be imported first in ``main.py`` to ensure a consistent log
format across every sub-module before any other logger is instantiated.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a standardised format.

    Sets up a single ``StreamHandler`` on ``sys.stdout`` so that all log
    records from any module in the project share the same format and respect
    the requested verbosity level.

    Args:
        level: A standard logging level string, e.g. ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.

    Raises:
        ValueError: If ``level`` is not a valid logging level name.

    Example:
        >>> from src.utils.logging_config import setup_logging
        >>> setup_logging("DEBUG")
    """
    numeric_level: int = getattr(logging, level.upper(), None)  # type: ignore[arg-type]
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: '{level}'")

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
        force=True,  # Override any handlers set by third-party libs (e.g. Hydra)
    )
