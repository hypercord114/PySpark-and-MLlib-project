import os
import logging
import json
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

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

file_handler = logging.FileHandler(os.path.join(LOG_DIR, '04_train_classification.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def train_and_evaluate(train_df, test_df, model_name, classifier, config):
    # 1. Define Pipeline
    indexer = StringIndexer(inputCol="Segment_Label", outputCol="label")
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features_vector")
    scaler = StandardScaler(inputCol="features_vector", outputCol="features")
    pipeline = Pipeline(stages=[indexer, assembler, scaler, classifier])

    # 2. Calibration (Param Grid)
    config_key = model_name.lower()
    if hasattr(classifier, "maxDepth"):
        params = config[config_key]
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.maxDepth, params["max_depth"]) \
            .addGrid(classifier.numTrees, params["num_trees"]) \
            .build()
    else:
        paramGrid = ParamGridBuilder().build()
    
    # 3. Cross-Validation
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # 4. MLflow Loop
    with mlflow.start_run(run_name=model_name):
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        
        # Log Metrics
        acc = evaluator.evaluate(best_model.transform(test_df))
        mlflow.log_metric("accuracy", acc)
        mlflow.spark.log_model(best_model, "model")
        mlflow.log_artifact("config.json")
        logger.info(f"{model_name} completed. Accuracy: {acc:.4f}")

def main():
    spark = SparkSession.builder.appName("PredictSegments").getOrCreate()
    logger.info("Loading and splitting data...")
    df = spark.read.parquet(LABELED_DATA)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label"),
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label")
    }

    for name, clf in models.items():
        logger.info(f"Starting experiment for: {name}")
        train_and_evaluate(train_df, test_df, name, clf)

    spark.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")