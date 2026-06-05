#!/bin/bash

# 1. Ensure the raw_data directory exists
mkdir -p raw_data

# 2. Check if csvkit is installed
if ! command -v in2csv &> /dev/null; then
    echo "csvkit not found. Installing..."
    pip3 install --user csvkit
    # Add local bin to path if needed for the current session
    export PATH="$PATH:$HOME/.local/bin"
fi

# 3. Check if there are any xlsx files in the directory
xlsx_files=(*.xlsx)
if [ ! -e "${xlsx_files[0]}" ]; then
    echo "No .xlsx files found in the current directory."
    exit 0
fi

# 4. Process files and move them to raw_data/
echo "Starting conversion..."
for file in *.xlsx; do
    output_name="raw_data/${file%.xlsx}.csv"
    echo "Converting: $file -> $output_name"
    in2csv "$file" > "$output_name"
done

echo "All files have been converted and saved to the 'raw_data/' directory."