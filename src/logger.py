import os
import logging
from datetime import datetime

# Generate the log file name
log_file = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

# Define the directory for logs
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists

# Full path to the log file
log_path_file = os.path.join(log_dir, log_file)

# Configure logging
logging.basicConfig(
    filename=log_path_file,
    format='[%(asctime)s]-%(name)s-%(lineno)d-%(levelname)s-%(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging started")
    logging.info("Logging ended")







