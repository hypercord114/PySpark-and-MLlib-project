import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import logging
import mlflow
import mlflow.prophet
import pandas as pd
from prophet import Prophet
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

print(f"DEBUG: Current Tracking URI is: {mlflow.get_tracking_uri()}")
print(f"DEBUG: Environment variable is: {os.getenv('MLFLOW_TRACKING_URI')}")

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
TS_DATA_PATH = os.path.join(BASE_DIR, "data/features/revenue_timeseries.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models/revenue_forecast")

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

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

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast_data.csv", index=False)

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RevenueForecast").getOrCreate()
    try:
        run_forecast(spark)
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
    finally:
        spark.stop()