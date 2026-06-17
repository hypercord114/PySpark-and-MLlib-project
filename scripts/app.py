import streamlit as st
import pandas as pd
import os
import mlflow
from mlflow.tracking import MlflowClient
import plotly.express as px

# Set page layout
st.set_page_config(layout="wide")

st.title("Customer Segmentation Dashboard [Jan. 1 2012]")
st.header(" - Unsupervised ML clustering of customers based on transaction history -")

# Paths
ANALYTICS_DIR = os.path.join(os.getcwd(), 'data/analytics')
summary_path = os.path.join(ANALYTICS_DIR, "segment_summary.parquet")
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(root_dir, "mlruns.db")

mlflow.set_tracking_uri(f"sqlite:///{db_path}")

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

@st.cache_data
def get_forecast():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    file_path = os.path.join(root_dir, "forecast_data.csv")

    return pd.read_csv(file_path)

@st.cache_resource
def get_mlflow_data():
    return mlflow.search_runs(experiment_ids=["0"])

if os.path.exists(summary_path):
    # Load your aggregated data
    df = pd.read_parquet(summary_path)
    df = df.sort_values(by="Customer_Segment", ascending=True)
    
    st.subheader("Customer Segment Profiles (finances and executive decisions may determine segment labels)")
    st.dataframe(df, hide_index=True)

    # Simple chart
    st.subheader("Average Monetary Spend by Customer Segment")
    st.bar_chart(df.set_index('Customer_Segment')['Avg_Monetary'])

    # Revenue forecast model predictions
    st.header(" - Supervised ML prediction model Revenue Forecast projection -")
    forecast_df = get_forecast()
    st.line_chart(forecast_df.set_index('ds')[['yhat']])

    # Experiment data
    st.header(" - Assessment of Unsupervised ML clustering & Supervised ML prediction model training -")

    # - Get all runs
    runs = get_mlflow_data()

    if runs_df is not None and not runs_df.empty:
        # --- Part A: Display the Summary ---
        cols_to_drop = [col for col in runs_df.columns if "log-model.history" in col]
        runs_clean = runs_df.drop(columns=cols_to_drop, errors='ignore')
    
        st.write("### Experiment Data")
        st.dataframe(runs_clean)

        # --- Part B: Fetch Detailed Metric History for Plots ---
        all_steps = []
        client = mlflow.tracking.MlflowClient()

        # Iterate through the DataFrame rows to get the Run IDs
        for _, run in runs_df.iterrows():
            run_id = run['run_id']
            run_name = run.get("tags.mlflow.runName", "Unknown")
        
            # Fetch metric history for specific metrics
            for metric_name in ["silhouette_score", "accuracy", "auc"]:
                try:
                    history = client.get_metric_history(run_id, metric_name)
                    for m in history:
                        all_steps.append({
                            "run_name": run_name,
                            "metric": metric_name,
                            "step": m.step,
                            "value": m.value
                        })
                except:
                    continue # Metric might not exist for this run

        # 3. Create the detailed DataFrame for Plotly
        detailed_df = pd.DataFrame(all_steps)
    
        # 4. Filter and plot using the new detailed_df
        def get_plot_data(df, metric_name):
            return df[df['metric'] == metric_name]

        st.plotly_chart(plot_model_metric(get_plot_data(detailed_df, "silhouette_score"), "silhouette_score"))
        st.plotly_chart(plot_model_metric(get_plot_data(detailed_df, "accuracy"), "accuracy"))
        st.plotly_chart(plot_model_metric(get_plot_data(detailed_df, "auc"), "auc"))

    else:
        st.warning("No runs found in Experiment 0.")

else:
    st.error("Analytics data not found. Please run your 04_analyze_segments.py script first!")
