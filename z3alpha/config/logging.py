"""App logging: :func:`setup_logging` for CLIs; :func:`attach_file_logger` for MCTS trace files."""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT)


def attach_file_logger(
    name: str,
    log_file: str | Path,
    *,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Single file, no root propagation; replacing handlers on repeat calls in-process."""
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
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr:
            return True
    return False


def setup_logging(level: str | int = "INFO") -> None:
    """Root level + one stderr :class:`StreamHandler` with :data:`LOG_FORMAT` (idempotent)."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not _root_has_stderr_handler(root):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(get_formatter())
        root.addHandler(h)
