"""
NL2SQL Logging Module
Centralized logging configuration for the entire project

Author: Pranav
Date: 2026-04-02
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import LOGS, MAIN_LOG


def get_logger(
    name: str = "nl2sql",
    log_file: Path = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (usually module name)
        log_file: Optional custom log file path
        console_level: Logging level for console output
        file_level: Logging level for file output

    Returns:
        Configured logger instance

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Message")
        logger.error("Error occurred")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (colored output for terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = log_file or MAIN_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log separator on startup
    logger.info("="*80)
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log file: {log_path}")
    logger.info("="*80)

    return logger


def setup_module_logger(module_name: str) -> logging.Logger:
    """
    Convenience function to get a logger for a specific module.
    Automatically uses module's __name__.

    Args:
        module_name: Usually __name__ from the calling module

    Returns:
        Configured logger instance
    """
    return get_logger(module_name)


# ============================================================================
# CUSTOM FILTERS
# ============================================================================

class MaxLevelFilter(logging.Filter):
    """Filter to limit maximum logging level."""

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


# ============================================================================
# CONTEXT MANAGER FOR LOGGING
# ============================================================================

class LogContext:
    """
    Context manager for timing and logging operations.

    Usage:
        with LogContext(logger, "Processing data"):
            # do work
            pass
    """

    def __init__(self, logger: logging.Logger, operation: str, log_level: int = logging.INFO):
        self.logger = logger
        self.operation = operation
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.log_level, f"▶ Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.log(
                self.log_level,
                f"[OK] Completed: {self.operation} (duration: {duration})"
            )
        else:
            self.logger.error(
                f"[FAIL] Failed: {self.operation} (duration: {duration}) - {exc_val}"
            )


# ============================================================================
# TEST FUNCTION
# ============================================================================

def _test_logger():
    """Test the logger configuration."""
    logger = get_logger("test_logger")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    with LogContext(logger, "Test operation"):
        import time
        time.sleep(0.5)

    print(f"\n[OK] Check log file at: {MAIN_LOG}")


if __name__ == "__main__":
    _test_logger()
