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

def plot_model_metric(df, metric_name):

    col_name = f"metrics.{metric_name}"
    
    chart_df = df[['tags.mlflow.runName', col_name]].dropna()
    chart_df.columns = ['Model', 'Score']
    
    fig = px.bar(
        chart_df, 
        x='Score', 
        y='Model', 
        orientation='h',
        title=f"Comparison by {metric_name.capitalize()}",
        color='Score',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(template="plotly_dark", yaxis={'categoryorder': 'total ascending'})
    return fig

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
    st.header(" - Assessment of Unsupervised ML clustering & Supervised ML prediction model training -")
    runs = mlflow.search_runs(experiment_names=["Default"])
    st.write("### Experiment Data")
    st.dataframe(runs)

    # Charts
    st.plotly_chart(plot_model_metric(runs, "silhouette_score"), use_container_width=True)
    st.plotly_chart(plot_model_metric(runs, "accuracy"), use_container_width=True)
    st.plotly_chart(plot_model_metric(runs, "auc"), use_container_width=True)

else:
    st.error("Analytics data not found. Please run your 04_analyze_segments.py script first!")
