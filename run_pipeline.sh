#!/bin/bash

# Exit immediately if any command fails
set -e

echo "--- Cleaning up old data ---"
sudo rm -rf data/

echo "--- Building Docker Image ---"
docker build -t rfm-pipeline .

echo "--- Starting Pipeline Execution ---"
docker run --rm \
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