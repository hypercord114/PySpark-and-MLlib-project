#!/bin/bash

# Exit immediately if any command fails
set -e

echo "--- Cleaning up old data ---"
sudo rm -rf data/
sudo rm -rf logs/
sudo rm -rf models/
sudo rm -rf mlflow.db
sudo rm -rf mlruns.db
sudo rm -rf mlruns/
sudo rm -rf forecast_data.csv

echo "--- Building Docker Image ---"
docker build -t rfm-pipeline .

echo "--- Starting Pipeline Execution ---"
docker run --rm \
  -p 5000:5000 \
  -e HOME=/tmp \
  -e USER=sparkuser \
  -e IVY_HOME=/tmp/.ivy2 \
  -e SPARK_LOCAL_DIRS=/tmp/spark \
  -e JAVA_TOOL_OPTIONS="-Duser.home=/tmp" \
  -v "$(pwd):/app" \
  -w /app \
  rfm-pipeline python3 scripts/main_orchestrator.py

echo "--- Pipeline Complete! ---"

# List the generated files for verification
ls -R data/

echo "--- Fixing file permissions ---"
sudo chown -R "$(id -u):$(id -g)" data/

echo "--- Ensuring MLflow is updated ---"
pip install --upgrade "mlflow>=3.10" > /dev/null

if [ -d "mlruns" ]; then
    echo "--- Migrating file-based metadata to mlruns.db ---"
    python3 -m mlflow migrate-filestore --source ./mlruns --target sqlite:///mlruns.db
else
    echo "Warning: ./mlruns directory not found, skipping migration."
fi

echo "--- All tasks finished! ---"