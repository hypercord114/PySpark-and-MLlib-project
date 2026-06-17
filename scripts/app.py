import streamlit as st
import pandas as pd
import os
import mlflow
import plotly.express as px
from mlflow.tracking import MlflowClient

# Set page layout
st.set_page_config(layout="wide")

st.title("Customer Segmentation Dashboard [Jan. 1 2012]")
st.header(" - Unsupervised ML clustering of customers based on transaction history -")

# --- PATH CONFIGURATION ---
ANALYTICS_DIR = os.path.join(os.getcwd(), 'data/analytics')
summary_path = os.path.join(ANALYTICS_DIR, "segment_summary.parquet")
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(root_dir, "mlruns.db")

# --- MLFLOW SETUP ---
@st.cache_resource
def setup_mlflow():
    # Detect if we are on Streamlit Cloud (read-only environment)
    if "STREAMLIT_SERVER_PORT" in os.environ:
        mlruns_path = os.path.join(root_dir, "mlruns")
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
    else:
        # Use DB locally with Read-Only mode to avoid locks
        mlflow.set_tracking_uri(f"sqlite:///{db_path}?mode=ro")
    return MlflowClient()

client = setup_mlflow()

# --- HELPER FUNCTIONS ---
def plot_model_metric(df, metric_name):
    # This expects a DataFrame with 'Model' and 'Score' columns
    fig = px.bar(
        df, x='Score', y='Model', orientation='h',
        title=f"Comparison by {metric_name.capitalize()}",
        color='Score', color_continuous_scale='Blues'
    )
    fig.update_layout(template="plotly_dark", yaxis={'categoryorder': 'total ascending'})
    return fig

@st.cache_data
def get_mlflow_data():
    return mlflow.search_runs(experiment_ids=["0"])

# --- EXISTING CONTENT ---
if os.path.exists(summary_path):
    df = pd.read_parquet(summary_path)
    df = df.sort_values(by="Customer_Segment", ascending=True)
    
    st.subheader("Customer Segment Profiles")
    st.dataframe(df, hide_index=True)
    st.bar_chart(df.set_index('Customer_Segment')['Avg_Monetary'])

    st.header(" - Supervised ML prediction model Revenue Forecast projection -")
    forecast_df = pd.read_csv(os.path.join(root_dir, "forecast_data.csv"))
    st.line_chart(forecast_df.set_index('ds')[['yhat']])

    # --- INTEGRATED EXPERIMENT DATA ---
    st.header(" - Assessment of ML training -")
    runs = get_mlflow_data()
    
    if runs is not None and not runs.empty:
        # Clean data for display
        cols_to_drop = [col for col in runs.columns if "log-model.history" in col]
        runs_clean = runs.drop(columns=cols_to_drop, errors='ignore')
        st.dataframe(runs_clean)

        # Prepare data for plotting
        for metric in ["silhouette_score", "accuracy", "auc"]:
            metric_col = f"metrics.{metric}"
            if metric_col in runs.columns:
                plot_df = runs[['tags.mlflow.runName', metric_col]].dropna()
                plot_df.columns = ['Model', 'Score']
                st.plotly_chart(plot_model_metric(plot_df, metric), use_container_width=True)
    else:
        st.warning("No runs found in Experiment 0.")
else:
    st.error("Analytics data not found.")