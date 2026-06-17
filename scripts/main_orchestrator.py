import os
import logging
import subprocess

# Ensure logs directory exists
BASE_DIR = "/app"

# Define all other paths relative to the absolute base
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Configure logging to write to both console and file
logger = logging.getLogger("Orchestrator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler for /logs/orchestration.log
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'orchestration.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def run_pipeline():
    tasks = [
            ["python3", "-u", "scripts/00_download_and_prep.py"],
            ["python3", "-u", "scripts/01_clean_data.py"],
            ["python3", "-u", "scripts/02_engineer.py"],
            ["python3", "-u", "scripts/03_cluster.py"],
            ["python3", "-u", "scripts/04_analyze_segments.py"],
            ["python3", "-u", "scripts/05_train_classification.py"],
            ["python3", "-u", "scripts/06_train_churn.py"],
            ["python3", "-u", "scripts/07_forecast_revenue.py"]
        ]

    for task in tasks:
        logger.info(f"Starting task: {' '.join(task)}")
        try:
            subprocess.run(task, check=True)
            logger.info(f"Finished: {' '.join(task)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed at {' '.join(task)} with error: {e}")

            break

if __name__ == "__main__":
    run_pipeline()