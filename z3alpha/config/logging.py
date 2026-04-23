"""
Central logging configuration for z3alpha.
Call setup_logging() from CLI entry points (synthesize, scripts, etc.).
Library modules should use logging.getLogger(__name__); only this module
should attach handlers (except attach_file_logger for trace files).
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_formatter():
    """Return the standard formatter for z3alpha log handlers."""
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT)


def attach_file_logger(
    name: str,
    log_file: str | Path,
    *,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Logger that writes only to one file (no propagation to the root logger).
    Replaces any existing handlers on the same logger name so repeated runs
    in one process do not duplicate output.
    """
    path = Path(log_file)
    lg = logging.getLogger(name)
    lg.propagate = False
    for handler in lg.handlers[:]:
        lg.removeHandler(handler)
        handler.close()
    lg.setLevel(level)
    fh = logging.FileHandler(path)
    fh.setFormatter(get_formatter())
    lg.addHandler(fh)
    return lg


def _root_has_stderr_handler(root: logging.Logger) -> bool:
    """True if root already has a handler writing to sys.stderr (not e.g. a file)."""
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr:
            return True
    return False


def setup_logging(level="INFO"):
    """
    Configure logging for the application. Safe to call more than once.
    Sets the root logger level and ensures one stderr StreamHandler with the
    standard z3alpha format.

    level: log level name (e.g. "INFO", "DEBUG") or logging constant.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not _root_has_stderr_handler(root):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(get_formatter())
        root.addHandler(handler)
