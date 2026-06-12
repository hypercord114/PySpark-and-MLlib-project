import os
import logging
import mlflow
import mlflow.prophet
import pandas as pd
from prophet import Prophet
from pyspark.sql import SparkSession

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
TS_DATA_PATH = os.path.join(BASE_DIR, "data/features/revenue_timeseries.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models/revenue_forecast")

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(os.path.join(LOG_DIR, '06_forecast_revenue.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def run_forecast(spark):
    logger.info("Loading time-series data...")
    # Load the revenue timeseries created by 02_engineer.py
    df = spark.read.parquet(TS_DATA_PATH)
    
    # 1. Prepare data for Prophet
    # Prophet requires columns named 'ds' (date) and 'y' (value)
    pdf = df.select(col("Date").alias("ds"), col("Weekly_Revenue").alias("y")).toPandas()
    
    # 2. Train Prophet Model
    logger.info("Training Prophet model...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(pdf)
    
    # 3. Log to MLflow
    with mlflow.start_run(run_name="Revenue_Forecast"):
        # Log the model artifact
        mlflow.prophet.log_model(model, "model")
        
        # Log basic metadata
        mlflow.log_param("model_type", "Prophet")
        logger.info("Prophet model logged to MLflow successfully.")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RevenueForecast").getOrCreate()
    try:
        run_forecast(spark)
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
    finally:
        spark.stop()