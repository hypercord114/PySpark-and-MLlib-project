import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import logging
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, unix_date

print(f"DEBUG: Current Tracking URI is: {mlflow.get_tracking_uri()}")
print(f"DEBUG: Environment variable is: {os.getenv('MLFLOW_TRACKING_URI')}")

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
CHURN_DATA = os.path.join(BASE_DIR, "data/features/churn_features.parquet")

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(os.path.join(LOG_DIR, '06_train_churn.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def train_churn_model(spark):
    logger.info("Loading churn features...")
    df = spark.read.parquet(CHURN_DATA)
    
    # Define features and label
    # Note: 'is_churned' is already 0 or 1 from your 02_engineer.py script
    df = df.withColumn("LastPurchase_numeric", unix_date(col("LastPurchase")).cast("double"))
    assembler = VectorAssembler(inputCols=["LastPurchase_numeric"], outputCol="features")
    rf = RandomForestClassifier(featuresCol="features", labelCol="is_churned")
    
    pipeline = Pipeline(stages=[assembler, rf])
    
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    with mlflow.start_run(run_name="Churn_Prediction"):
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        
        # Binary evaluation
        evaluator = BinaryClassificationEvaluator(labelCol="is_churned")
        auc = evaluator.evaluate(predictions)
        
        mlflow.log_metric("auc", auc)
        mlflow.spark.log_model(model, "churn_model")
        logger.info(f"Churn Model AUC: {auc}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()
    train_churn_model(spark)
    spark.stop()