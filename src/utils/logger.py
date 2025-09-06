"""
Logging utility for the predictive maintenance system.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any


def setup_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Logger name
        config: Logging configuration dictionary

    Returns:
        Configured logger instance
    """
    if config is None:
        config = {}

    logger = logging.getLogger(name)

    # Avoid adding handlers if logger already has them
    if logger.handlers:
        return logger

    # Set logging level
    level = getattr(logging, config.get('level', 'INFO').upper())
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Console handler
    console_config = config.get('handlers', {}).get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, console_config.get('level', 'INFO').upper())
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    file_config = config.get('handlers', {}).get('file', {})
    if file_config.get('enabled', False):
        log_dir = os.path.dirname(file_config.get('filename', 'logs/application.log'))
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(file_config['filename'])
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
