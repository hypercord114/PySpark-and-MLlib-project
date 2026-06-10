import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, max, datediff, lit, add_months, trunc

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed_data/cleaned_retail_data.parquet")
FEATURE_DIR = os.path.join(BASE_DIR, "data/features")

# Create directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# Configure logging to write to both file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'feature_engineering.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def engineer_features(spark, parquet_path):
    logger.info("Loading cleaned data for feature engineering...")
    df = spark.read.parquet(parquet_path)

    # 1. Calculate Monetary value per transaction
    df = df.withColumn("TotalSpend", col("Quantity") * col("UnitPrice"))

    # 2. Aggregating to Customer level (RFM)
    # * Reset date of report to first day of month following max invoice date
    global_max_date = df.select(max("InvoiceDate")).collect()[0][0]
    following_month_global = add_months(trunc(lit(global_max_date), "MM"), 1)
    logger.info(f"Faux date of report calculated to be: {following_month_global}")

    rfm_df = df.groupBy("CustomerID").agg(
        datediff(following_month_global, max("InvoiceDate")).alias("Recency"),
        count("InvoiceNo").alias("Frequency"),
        sum("TotalSpend").alias("Monetary")
    )
    
    return rfm_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    try:
        rfm_features = engineer_features(spark, PROCESSED_PATH)
        
        # Save the features
        output_path = os.path.join(FEATURE_DIR, "rfm_features.parquet")
        rfm_features.write.mode("overwrite").parquet(output_path)
        logger.info(f"RFM features saved to {output_path}")
        rfm_features.show(5)
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")