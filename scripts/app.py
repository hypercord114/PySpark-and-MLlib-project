import streamlit as st
import pandas as pd
import os
import mlflow
import plotly.express as px

# Set page layout
st.set_page_config(layout="wide")

st.title("Customer Segmentation Dashboard [Jan. 1 2012]")
st.header(" - Unsupervised ML clustering of customers based on transaction history -")

# Paths (adjust to where your script saved the parquet files)
ANALYTICS_DIR = os.path.join(os.getcwd(), 'data/analytics')
summary_path = os.path.join(ANALYTICS_DIR, "segment_summary.parquet")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

if os.path.exists(summary_path):
    # Load your aggregated data
    df = pd.read_parquet(summary_path)
    df = df.sort_values(by="Customer_Segment", ascending=True)
    
    st.subheader("Customer Segment Profiles (finances and executive decisions may determine segment labels)")
    st.dataframe(df, hide_index=True)

    # Simple chart
    st.subheader("Average Monetary Spend by Customer Segment")
    st.bar_chart(df.set_index('Customer_Segment')['Avg_Monetary'])

    # Experiment data
    runs = mlflow.search_runs(experiment_names=["Default"])
    st.write("### Experiment Data")
    st.dataframe(runs)

    # Charts
    df_metrics = runs[['tags.mlflow.runName', 'metrics.accuracy']].dropna()
    df_metrics.columns = ['Model', 'Accuracy']

    # Create the horizontal bar chart
    fig = px.bar(
        df_metrics, 
        x='Accuracy', 
        y='Model', 
        orientation='h',
        title="Model Accuracy Comparison",
        color='Accuracy',
        color_continuous_scale='Viridis'
    )

    # Style it to look like the MLflow UI
    fig.update_layout(
        xaxis_range=[0, 1],
        template="plotly_dark",
        yaxis={'categoryorder': 'total ascending'}
    )

    # Display
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Analytics data not found. Please run your 04_analyze_segments.py script first!")
