import subprocess
import logging

# Set up logging for the orchestrator
logging.basicConfig(level=logging.INFO)

def run_script(script_path):
    logging.info(f"Starting {script_path}...")
    # This runs the command exactly as you would in your terminal
    result = subprocess.run(["python3", script_path], check=True)
    logging.info(f"Finished {script_path} with return code {result.returncode}")

if __name__ == "__main__":
    try:
        # Define your rigid pipeline order
        run_script("scripts/01_clean.py")
        run_script("scripts/02_engineer.py")      #
        run_script("scripts/03_cluster.py")       #
        run_script("scripts/04_analyze_segments.py") #
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failed at {e.cmd}")
