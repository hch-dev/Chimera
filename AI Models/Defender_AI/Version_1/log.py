import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.INFO)

    # Rotating file handler keeps logs from growing too large
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=2_000_000,  # 2MB per log file
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # Prevent double logging

    return logger
