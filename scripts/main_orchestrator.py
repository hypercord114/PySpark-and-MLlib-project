import os
import subprocess
import logging
import mlflow

mlflow.set_tracking_uri("file:///app/mlruns")

# Ensure logs directory exists
BASE_DIR = "/app"

# Now define all other paths relative to the absolute base
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

def run_script(script_path):
    logger.info(f"Starting {script_path}...")
    # This runs the command exactly as you would in your terminal
    result = subprocess.run(["python3", script_path], check=True)
    logger.info(f"Finished {script_path} with return code {result.returncode}")

if __name__ == "__main__":
    try:
        # Define your rigid pipeline order
        run_script("scripts/00_download_and_prep.py")
        run_script("scripts/01_clean_data.py")
        run_script("scripts/02_engineer.py")
        run_script("scripts/03_cluster.py")
        run_script("scripts/04_analyze_segments.py")
        run_script("scripts/05_train_classification.py")
        run_script("scripts/06_train_churn.py")
        run_script("scripts/07_forecast_revenue.py")
        logger.info("Pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed at {e.cmd}")
