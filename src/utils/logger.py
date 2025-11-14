"""
Logging Configuration

Centralized logging setup for the framework.
Supports console and file logging with configurable levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


_loggers = {}  # Cache of configured loggers


def setup_logger(
    name: str = "witness_framework",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger.

    Args:
        name: Logger name
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        log_file: Specific log file name (default: timestamped)

    Returns:
        Configured logger
    """
    # Return cached logger if already configured
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    # Format: [TIME] [LEVEL] [MODULE] MESSAGE
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path('logs')
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Cache logger
    _loggers[name] = logger

    logger.info(f"Logger '{name}' initialized with level {level}")

    return logger


def get_logger(name: str = "witness_framework") -> logging.Logger:
    """
    Get existing logger or create default one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    else:
        return setup_logger(name)
