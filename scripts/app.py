import streamlit as st
import pandas as pd
import os

# Set page layout
st.set_page_config(layout="wide")

st.title("Customer Segmentation Dashboard")
st.title(" - Unsupervised ML Clustering of customers -")

# Paths (adjust to where your script saved the parquet files)
ANALYTICS_DIR = os.path.join(os.getcwd(), 'data/analytics')
summary_path = os.path.join(ANALYTICS_DIR, "segment_summary.parquet")

if os.path.exists(summary_path):
    # Load your aggregated data
    df = pd.read_parquet(summary_path)
    df = df.sort_values(by="customer segment", ascending=True)
    
    st.subheader("Cluster Profiles")
    st.dataframe(df)

    # Simple chart
    st.subheader("Average Monetary Spend by Cluster")
    st.bar_chart(df.set_index('prediction')['Avg_Monetary'])
else:
    st.error("Analytics data not found. Please run your 04_analyze_segments.py script first!")