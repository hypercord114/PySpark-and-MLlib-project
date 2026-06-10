#!/bin/bash

# Exit immediately if any command fails
set -e

echo "--- Building Docker Image ---"
docker build -t rfm-pipeline .

echo "--- Starting Pipeline Execution ---"
# -v $(pwd):/app mounts your current directory to the container
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd):/app" rfm-pipeline python3 scripts/main_orchestrator.py

echo "--- Pipeline Complete! ---"
# List the generated files for verification
ls -R data/