#!/bin/bash

# Define the target directory
DATA_DIR="../data"
ZIP_FILE="$DATA_DIR/online_retail.zip"
URL="https://archive.ics.uci.edu/static/public/352/online+retail.zip"

echo "Checking if data directory exists..."
mkdir -p "$DATA_DIR"

echo "Downloading dataset..."
# Use -L to follow redirects and -o to specify output file
curl -L "$URL" -o "$ZIP_FILE"

echo "Unzipping dataset..."
unzip "$ZIP_FILE" -d "$DATA_DIR"

echo "Cleaning up zip file..."
rm "$ZIP_FILE"

echo "Download and extraction complete!"