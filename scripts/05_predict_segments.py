import os
import logging
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
LABELED_DATA = os.path.join(BASE_DIR, "data/analytics/labeled_customers.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models/segment_classifier")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'model_training.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def train_predictor(spark):
    logger.info("Loading labeled customer data...")
    df = spark.read.parquet(LABELED_DATA)

    # 1. Convert string labels to numbers
    indexer = StringIndexer(inputCol="Segment_Label", outputCol="label")
    
    # 2. Assemble and Scale features
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    # 3. Define the Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")

    # 4. Create the Pipeline
    pipeline = Pipeline(stages=[indexer, assembler, scaler, rf])

    # 5. Split Data
    logger.info("Splitting data for training and evaluation...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # 6. Train
    logger.info("Training Random Forest model...")
    model = pipeline.fit(train_df)

    # 7. Evaluate
    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    # Log the result instead of printing
    logger.info(f"Model Training Complete. Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    model.write().overwrite().save(MODEL_DIR)
    logger.info(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("PredictSegments").getOrCreate()
    try:
        train_predictor(spark)
    except Exception as e:
        logger.error(f"Training failed: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")