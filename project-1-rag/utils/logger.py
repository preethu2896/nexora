"""Logging utility for structured and clean console logging."""

import logging
import sys
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for enhanced CLI readability."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: blue + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.fmt)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger(name: str, log_level: Optional[str] = "INFO") -> logging.Logger:
    """Configures and returns a logger instance with custom formatting.

    Args:
        name: The module or class name for log context.
        log_level: Desired log level string (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        numeric_level = getattr(logging, log_level.upper() if log_level else "INFO", logging.INFO)
        logger.setLevel(numeric_level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(console_handler)
        logger.propagate = False

    return logger
