import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, max, datediff, lit, add_months, trunc, when

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
file_handler = logging.FileHandler(os.path.join(LOG_DIR, '02_engineer.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def engineer_features(spark, parquet_path):
    logger.info("Loading cleaned data for feature engineering...")
    df = spark.read.parquet(parquet_path)
    df = df.withColumn("TotalSpend", col("Quantity") * col("UnitPrice"))

    # --- RFM Features (For Segmentation) ---
    # * Reset date of report to first day of month following max invoice date
    global_max_date = df.select(max("InvoiceDate")).collect()[0][0]
    following_month_global_expr = add_months(trunc(lit(global_max_date), "MM"), 1)
    following_month_global = spark.range(1).select(following_month_global_expr).collect()[0][0]
    logger.info(f"Faux date of report calculated to be: {following_month_global}")

    rfm_df = df.groupBy("CustomerID").agg(
        datediff(following_month_global_expr, max("InvoiceDate")).alias("Recency"),
        count("InvoiceNo").alias("Frequency"),
        sum("TotalSpend").alias("Monetary")
    )
    # Save RFM feature parquet
    rfm_df.write.mode("overwrite").parquet(os.path.join(FEATURE_DIR, "rfm_features.parquet"))
    logger.info("RFM feature parquet saved successfully.")
    
    # --- Churn Features (Binary Classification) ---
    # * Churn = no purchase in 60 days
    churn_df = df.groupBy("CustomerID").agg(
        max("InvoiceDate").alias("LastPurchase")
    ).withColumn(
        "is_churned", 
        when(datediff(following_month_global_expr, col("LastPurchase")) > 60, 1).otherwise(0)
    )
    # Save churn feature parquet
    churn_df.write.mode("overwrite").parquet(os.path.join(FEATURE_DIR, "churn_features.parquet"))
    logger.info("Churn feature parquet saved successfully.")

    # --- Revenue Time-Series ---
    ts_df = df.withColumn("Date", trunc("InvoiceDate", "week")) \
              .groupBy("Date") \
              .agg(sum("TotalSpend").alias("Weekly_Revenue")) \
              .orderBy("Date")
    # Save revenue time series parquet
    ts_df.write.mode("overwrite").parquet(os.path.join(FEATURE_DIR, "revenue_timeseries.parquet"))
    logger.info("Revenue time series parquet saved successfully.")

    logger.info("All feature sets saved successfully.")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()
    try:
        rfm_features = engineer_features(spark, PROCESSED_PATH)
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")