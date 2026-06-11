import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, col, when

# Paths
BASE_DIR = "/app"
CLUSTER_DIR = os.path.join(BASE_DIR, "data/clusters")
ANALYTICS_DIR = os.path.join(BASE_DIR, "data/analytics")
os.makedirs(ANALYTICS_DIR, exist_ok=True)

def generate_bi_data(spark):
    # Load the results from your clustering script
    df = spark.read.parquet(os.path.join(CLUSTER_DIR, "customer_segments.parquet"))

    # Rename 'prediction' to 'customer segment'
    df = df.withColumnRenamed("prediction", "Customer_Segment")

    # Add Segment_Label to every individual customer record & save parquet
    df.withColumn(
        "Segment_Label",
        when((col("Avg_Recency") < 30) & (col("Avg_Frequency") >= 5) & (col("Avg_Monetary") >= 8000), "Low R, High F, High M - Champion - Reward them, ask for referrals, give early access to new products.")
        .when((col("Avg_Recency") >= 90) & (col("Avg_Frequency") < 30) & (col("Avg_Monetary") < 1000), "High R, Low F, Low M - At-Risk/Churned - Use win-back campaigns, surveys, or heavy discounts to re-engage.")
        .when((col("Avg_Recency") < 30) & (col("Avg_Frequency") < 30) & (col("Avg_Monetary") < 1000), "Low R, Low F, Low M - New Customer - Nurture them with welcome content, explain the brand value.")
        .when((col("Avg_Recency") >= 30) & (col("Avg_Frequency") >= 5) & (col("Avg_Monetary") >= 8000), "High R, High F, High M - Loyal/High Value - Keep them satisfied; personalize their experience so they stay.")
        .otherwise("Standard"))
    df.write.mode("overwrite").parquet(os.path.join(ANALYTICS_DIR, "labeled_customers.parquet"))

    # 1. Create Aggregated Summary for High-Level KPI cards
    summary_df = df.groupBy("Customer_Segment").agg(
        avg("Recency").alias("Avg_Recency"),
        avg("Frequency").alias("Avg_Frequency"),
        avg("Monetary").alias("Avg_Monetary"),
        count("CustomerID").alias("Customer_Count")
    )

    # 2. Add 'Segment_Label' using PySpark's conditional logic
    labeled_summary = summary_df.withColumn(
        "Segment_Label",
        when((col("Avg_Recency") < 30) & (col("Avg_Frequency") >= 5) & (col("Avg_Monetary") >= 8000), "Low R, High F, High M - Champion - Reward them, ask for referrals, give early access to new products.")
        .when((col("Avg_Recency") >= 90) & (col("Avg_Frequency") < 30) & (col("Avg_Monetary") < 1000), "High R, Low F, Low M - At-Risk/Churned - Use win-back campaigns, surveys, or heavy discounts to re-engage.")
        .when((col("Avg_Recency") < 30) & (col("Avg_Frequency") < 30) & (col("Avg_Monetary") < 1000), "Low R, Low F, Low M - New Customer - Nurture them with welcome content, explain the brand value.")
        .when((col("Avg_Recency") >= 30) & (col("Avg_Frequency") >= 5) & (col("Avg_Monetary") >= 8000), "High R, High F, High M - Loyal/High Value - Keep them satisfied; personalize their experience so they stay.")
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