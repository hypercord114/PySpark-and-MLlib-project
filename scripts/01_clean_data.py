import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col

# Setting paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_PATH = os.path.join(BASE_DIR, "data/raw_data/Online Retail.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed_data")

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Configure logging to write to a file AND terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'data_cleaning.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def get_retail_schema():
    return StructType([
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", DoubleType(), True),
        StructField("CustomerID", StringType(), True),
        StructField("Country", StringType(), True)
    ])

def clean_retail_data(spark, input_path):
    logger.info(f"Starting data cleaning for: {input_path}")
    
    df = spark.read.csv(input_path, header=False, schema=get_retail_schema())
    
    # Cleaning steps
    initial_count = df.count()
    df = df.filter(col("Quantity") > 0).na.drop(subset=["CustomerID"]).dropDuplicates()
    df = df.withColumn("InvoiceDate", col("InvoiceDate").cast(TimestampType()))
    
    final_count = df.count()
    logger.info(f"Data cleaning complete. Dropped {initial_count - final_count} rows.")

    # Save as Parquet
    df.write.mode("overwrite").parquet(os.path.join(PROCESSED_DIR, "cleaned_retail_data.parquet"))
    logger.info(f"Saved cleaned data to Parquet format. Saved {final_count} rows from the original dataset.")

    return df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RetailDataCleaning").getOrCreate()
    try:
        cleaned_df = clean_retail_data(spark, DATA_PATH)
        cleaned_df.show(5)
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")
