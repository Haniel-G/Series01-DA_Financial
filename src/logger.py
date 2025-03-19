'''
This script aims to define a logger with a standard logging message configuration.
Logging messages are employed to track application behavior, errors, and events. They facilitate debugging, provide insight into program flow, and aid in monitoring and diagnosing issues. Logging enhances code quality, troubleshooting, and maintenance.
'''

'''
Importing the libraries.
'''

# Debugging and verbose.
import logging
from datetime import datetime

# File handling.
import os

# The filename for the log file, including the current date and time.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# The path to the directory where log files are stored.
logs_path = os.path.join(os.getcwd(), "logs")

# Create the directory if it doesn't exist.
os.makedirs(logs_path, exist_ok=True)

# The full path to the log file.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Create a logger
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)

# Checks if there are already configured handlers, to avoid duplication
if not logger.handlers:
    # Create a file handler to write logs to the file with detailed format
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(
        logging.Formatter(
            "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        )
    )

    # Create a console handler with a simple format (just the message)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Basic test to ensure the log appears
#logger.info("Logging initialized successfully.")