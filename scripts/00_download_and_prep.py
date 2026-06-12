import os
import requests
import zipfile
import logging
import pandas as pd
import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '../logs')
DATA_DIR = os.path.join(SCRIPT_DIR, '../data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Configure logging
logger = logging.getLogger("DownloadPrep")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(LOG_DIR, '00_download_and_prep.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def download_and_prep():
    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    zip_path = os.path.join(DATA_DIR, "online_retail.zip")

    # 1. Download
    logger.info(f"Downloading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status() # Raises error if download fails
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # 2. Unzip
    logger.info("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)

    # 3. Convert .xlsx to .csv
    logger.info("Converting Excel files to CSV...")
    xlsx_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    
    if not xlsx_files:
        logger.warning("No .xlsx files found to convert.")
        return

    for file in xlsx_files:
        file_name = os.path.basename(file).replace(".xlsx", ".csv")
        output_path = os.path.join(RAW_DATA_DIR, file_name)
        
        logger.info(f"Converting: {file} -> {output_path}")
        df = pd.read_excel(file)
        df.to_csv(output_path, index=False)
        # Optional: remove the original xlsx to keep data/ clean
        os.remove(file)

    logger.info("Data download and preparation complete.")

if __name__ == "__main__":
    try:
        download_and_prep()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)