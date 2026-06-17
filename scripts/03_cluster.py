import os
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import logging
import numpy as np
import plotly.graph_objects as go
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

print(f"DEBUG: Current Tracking URI is: {mlflow.get_tracking_uri()}")
print(f"DEBUG: Environment variable is: {os.getenv('MLFLOW_TRACKING_URI')}")

# Paths
BASE_DIR = "/app"
LOG_DIR = os.path.join(BASE_DIR, "logs")
FEATURE_PATH = os.path.join(BASE_DIR, "data/features/rfm_features.parquet")
CLUSTER_DIR = os.path.join(BASE_DIR, "data/clusters")

if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(LOG_DIR, '03_cluster.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def run_elbow_method(spark, input_path):
    logger.info("Preparing data for Elbow Method...")
    df = spark.read.parquet(input_path)

    # 1. Feature Assembly & Scaling
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features_raw")
    df_assembled = assembler.transform(df)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)
    df_scaled = scaler.fit(df_assembled).transform(df_assembled)

    # 2. Calculate Cost for k = 2 to 10
    costs = []
    k_values = range(2, 11)
    
    logger.info("Calculating WCSS for k=2 to 10...")

    # Start MLflow tracking for each iteration of k in the elbow method
    with mlflow.start_run(run_name="Customer_Clustering"):
        # Iterate through k_values for performance of elbow method
        for k in k_values:
            kmeans = KMeans(k=k, seed=42)
            model = kmeans.fit(df_scaled)
            cost = model.summary.trainingCost
            costs.append(cost)

            predictions = model.transform(df_scaled)
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)

            mlflow.log_metric("WCSS", cost, step=k)
            mlflow.log_metric("silhouette_score", silhouette, step=k)

            logger.info(f"k={k}, WCSS={cost}, Silhouette={silhouette}")

    # 3. Generate HTML Plot using Plotly
    fig = go.Figure(data=go.Scatter(x=list(k_values), y=costs, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method for Optimal k',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='WCSS (Training Cost)'
    )
    
    html_path = os.path.join(BASE_DIR, 'index.html')
    fig.write_html(html_path)
    logger.info(f"Elbow plot saved to {html_path}")

    return k_values, costs, df_scaled

def find_elbow_point(k_values, costs):

    logger.info("Calculating Optimal k from Elbow Method data...")

    # 1. Convert to numpy arrays for vector math
    coords = np.vstack((k_values, costs)).T
    first_point = coords[0]
    last_point = coords[-1]
    
    # 2. Line equation from first to last point
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    # 3. Calculate distance from each point to the line
    vec_from_first = coords - first_point
    scalar_proj = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    dist_vec = vec_from_first - vec_proj
    distances = np.sqrt(np.sum(dist_vec**2, axis=1))
    
    # 4. The elbow is the point with the maximum distance
    elbow_index = np.argmax(distances)
    return k_values[elbow_index]

def run_final_clustering(df_scaled, optimal_k):
    logger.info("Loading RFM features for clustering...")

    # Final K-Means Clustering
    logger.info("Training K-Means model...")
    kmeans = KMeans(k=optimal_k, seed=42)
    model = kmeans.fit(df_scaled)
    
    # Add cluster predictions back to the original DF
    clustered_df = model.transform(df_scaled)

    logger.info(f"Clustering complete for k={optimal_k}")
    
    return model, clustered_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Clustering").getOrCreate()
    try:
        k_values, costs, df_scaled = run_elbow_method(spark, FEATURE_PATH)
        optimal_k = find_elbow_point(list(k_values), costs)
        logger.info(f"Automatically detected Optimal k: {optimal_k}")
        model, results = run_final_clustering(df_scaled, optimal_k)

        with mlflow.start_run(run_name="Customer_Clustering_Final"):
            mlflow.spark.log_model(model, "clustering_model")
            mlflow.log_param("optimal_k", optimal_k)
            evaluator = ClusteringEvaluator()
            final_silhouette = evaluator.evaluate(results)
            mlflow.log_metric("final_silhouette_score", final_silhouette)
            logger.info(f"Final model silhouette score: {final_silhouette}")

        # Save the full results dataframe (includes RFM, Scaled features, and predictions)
        results.write.mode("overwrite").parquet(
            os.path.join(CLUSTER_DIR, "customer_segments.parquet")
        )
        logger.info("Clustering complete. Full results saved.")

    except Exception as e:
        logger.error(f"Error during elbow method execution: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")