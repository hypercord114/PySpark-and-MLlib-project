import os
import logging
import numpy as np
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '../logs')
FEATURE_PATH = os.path.join(SCRIPT_DIR, '../data/features/rfm_features.parquet')
CLUSTER_DIR = os.path.join(SCRIPT_DIR, '../data/clusters')
os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging Setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'clustering.log'))
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
    for k in k_values:
        kmeans = KMeans(k=k, seed=42)
        model = kmeans.fit(df_scaled)
        cost = model.summary.trainingCost
        costs.append(cost)
        logger.info(f"k={k}, WCSS={cost}")

    # 3. Generate HTML Plot using Plotly
    fig = go.Figure(data=go.Scatter(x=list(k_values), y=costs, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method for Optimal k',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='WCSS (Training Cost)'
    )
    
    html_path = os.path.join(PROJECT_ROOT, 'index.html')
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
    
    return clustered_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Clustering").getOrCreate()
    try:
        k_values, costs, df_scaled = run_elbow_method(spark, FEATURE_PATH)
        optimal_k = find_elbow_point(list(k_values), costs)
        logger.info(f"Automatically detected Optimal k: {optimal_k}")
        results = run_final_clustering(df_scaled, optimal_k)

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