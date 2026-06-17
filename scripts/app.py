import os
import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px

# Set page layout
st.set_page_config(layout="wide")

st.title("Customer Segmentation Dashboard [Jan. 1 2012]")
st.header(" - Unsupervised ML clustering of customers based on transaction history -")

# --- PATH CONFIGURATION ---
ANALYTICS_DIR = os.path.join(os.getcwd(), 'data/analytics')
summary_path = os.path.join(ANALYTICS_DIR, "segment_summary.parquet")
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- MLFLOW SETUP ---
# Point directly to the SQLite database in the root repository
db_path = os.path.join(root_dir, "mlruns.db")
# Using '?mode=ro' to ensure read-only access, which is required for Streamlit Cloud
mlflow.set_tracking_uri(f"sqlite:///{db_path}?mode=ro")

@st.cache_data
def get_mlflow_data():
    """Retrieve all runs from experiment 0 via the SQLite database."""
    try:
        return mlflow.search_runs(experiment_ids=["0"])
    except Exception as e:
        st.error(f"Error loading MLflow data from database: {e}")
        return pd.DataFrame()

def plot_model_metric(df, metric_name):
    """Plot metrics directly from search_runs DataFrame."""
    col_name = f"metrics.{metric_name}"
    if col_name not in df.columns:
        return None
    
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

@st.cache_data
def get_forecast():
    file_path = os.path.join(root_dir, "forecast_data.csv")
    return pd.read_csv(file_path)

# --- MAIN DASHBOARD CONTENT ---
if os.path.exists(summary_path):
    # Load aggregated segment data
    df = pd.read_parquet(summary_path)
    df = df.sort_values(by="Customer_Segment", ascending=True)
    
    st.subheader("Customer Segment Profiles")
    st.dataframe(df, hide_index=True)

    st.subheader("Average Monetary Spend by Customer Segment")
    st.bar_chart(df.set_index('Customer_Segment')['Avg_Monetary'])

    # Revenue forecast model predictions
    st.header(" - Supervised ML Revenue Forecast -")
    forecast_df = get_forecast()
    st.line_chart(forecast_df.set_index('ds')[['yhat']])

    # --- INTEGRATED EXPERIMENT DATA ---
    st.header(" - Assessment of ML training -")
    runs = get_mlflow_data()
    
    if not runs.empty:
        # Display table (drop internal artifacts columns)
        cols_to_drop = [col for col in runs.columns if "log-model.history" in col]
        st.dataframe(runs.drop(columns=cols_to_drop, errors='ignore'))

        # Metrics plots
        for metric in ["silhouette_score", "accuracy", "auc"]:
            fig = plot_model_metric(runs, metric)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No MLflow runs found. Ensure 'mlruns.db' exists in the root directory.")

else:
    st.error("Analytics data not found. Please run your 04_analyze_segments.py script first!")