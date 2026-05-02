from __future__ import annotations

"""
Logging configuration for the training pipeline.

Sets up both console and file logging with consistent formatting.
Replaces existing handlers to avoid duplicate logs in Jupyter/interactive environments.
"""

import logging
from pathlib import Path


def setup_logging(log_file_path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the training pipeline.
    
    Sets up dual output (console + file) with consistent formatting.
    Clears existing handlers to avoid duplicate logs in interactive environments.
    
    Args:
        log_file_path: Path to log file.
        level: Logging level (default: INFO).
        
    Returns:
        Configured root logger instance.
    """
    # Create log directory if needed
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(level)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Register handlers with root logger

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger