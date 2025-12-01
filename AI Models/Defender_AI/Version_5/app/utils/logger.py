import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name="defender_v5", log_file="logs/defender.log"):
    # 1. Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 2. Configure the Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Formatter: [Time] [Level] [Message]
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s')

        # Handler 1: File (Rotates at 5MB, keeps 3 backups)
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        
        # Handler 2: Console (So you can see it running)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger