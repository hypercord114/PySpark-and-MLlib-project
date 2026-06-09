import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, when

# Paths
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--master local[*] pyspark-shell"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_DIR = os.path.join(SCRIPT_DIR, '../data/clusters')
ANALYTICS_DIR = os.path.join(SCRIPT_DIR, '../data/analytics')
os.makedirs(ANALYTICS_DIR, exist_ok=True)

def generate_bi_data(spark):
    # Load the results from your clustering script
    df = spark.read.parquet(os.path.join(CLUSTER_DIR, "customer_segments.parquet"))

    # 1. Create Aggregated Summary for High-Level KPI cards
    summary_df = df.groupBy("prediction").agg(
        avg("Recency").alias("Avg_Recency"),
        avg("Frequency").alias("Avg_Frequency"),
        avg("Monetary").alias("Avg_Monetary"),
        count("CustomerID").alias("Customer_Count")
    )

    # 2. Add 'Segment_Label' using PySpark's conditional logic
    labeled_summary = summary_df.withColumn(
        "Segment_Label",
        when((col("Avg_Recency") < 30) & (col("Avg_Frequency") > 5), "Champions")
        .when(col("Avg_Recency") > 90, "At Risk")
        .otherwise("Standard")
    )

    # 3. Save both for Power BI
    labeled_summary.write.mode("overwrite").parquet(os.path.join(ANALYTICS_DIR, "segment_summary.parquet"))
    # Keep the raw data for 'drill-down' features in Power BI
    df.write.mode("overwrite").parquet(os.path.join(ANALYTICS_DIR, "customer_drilldown.parquet"))

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Analytics").getOrCreate()
    generate_bi_data(spark)
    spark.stop()