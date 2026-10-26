"""
Logging configuration utility for ResumeMatch AI.
Provides clean console output and rotating file logging.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logger(
    name: str = "resumematch",
    log_file: Optional[str] = "app.log",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a structured logger.

    Args:
        name: Name of the logger instance.
        log_file: Log file name inside the logs directory, or None to skip file logging.
        level: Logging level (e.g., 'INFO', 'DEBUG'). Defaults to settings.LOG_LEVEL.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level_str = level or settings.LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatter for log records
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        settings.ensure_directories()
        log_path = settings.LOG_DIR / log_file
        try:
            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as exc:
            logger.warning("Could not attach file handler to %s: %s", log_path, exc)

    return logger


def get_logger(name: str = "resumematch") -> logging.Logger:
    """Retrieve or create a logger with standard configuration."""
    return setup_logger(name=name)
