"""
Central logging configuration for z3alpha.
Call setup_logging() from application entry points (e.g. synthesize.main()).
Library modules should only use logging.getLogger(__name__) and not add handlers.
"""

import logging

LOG_FORMAT = "%(asctime)s:%(levelname)s:%(name)s:%(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_formatter():
    """Return the standard formatter for z3alpha log handlers."""
    return logging.Formatter(LOG_FORMAT, datefmt=DATE_FMT)


def setup_logging(level="INFO"):
    """
    Configure logging for the application. Call once from an entry point (e.g. main()).
    Sets the root logger level and adds a single StreamHandler so that all
    z3alpha.* loggers emit to stderr with a consistent format.

    level: log level name (e.g. "INFO", "DEBUG") or logging constant.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if setup_logging is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(get_formatter())
        root.addHandler(handler)
