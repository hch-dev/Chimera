import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# --- Filepath Fix: Uses __file__ (double underscores) ---
# This correctly determines the directory where this script is located.
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure logs directory exists before logging starts
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with rotating file and console handlers.
    """
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if the logger is accessed multiple times
    if logger.handlers:
        return logger

    # Set logger level to DEBUG so all messages are passed up to the handlers
    logger.setLevel(logging.DEBUG)

    # --- File Handler (Rotating) ---
    # Log detailed information to a file, rotating it after 2MB.
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=2_000_000,  # 2MB per log file
        backupCount=5
    )
    file_handler.setLevel(logging.INFO) # Only INFO messages and above are written to the file

    # --- Console Handler ---
    # Log critical information to the console for real-time monitoring.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) # Show DEBUG messages in the console

    # Define the consistent output format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Disable propagation to the root logger to prevent duplicate output
    logger.propagate = False

    return logger

# --- EXPORTED LOGGER VARIABLE (Fixes 'cannot import name LOG') ---
# This makes the pre-configured logger available via 'from log import LOG'
LOG = get_logger('DefenderV2.Main')
