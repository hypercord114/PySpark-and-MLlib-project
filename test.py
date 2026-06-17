import mlflow
from mlflow.tracking import MlflowClient

# 1. Ensure the URI is set
mlflow.set_tracking_uri("sqlite:///mlruns.db")

# 2. Safely search for runs
try:
    runs = mlflow.search_runs(experiment_ids=["0"])
    print(f"Successfully retrieved {len(runs)} runs from experiment 0.")
except Exception as e:
    print(f"Could not retrieve runs: {e}")

# 3. Corrected loop to print experiment details
print("\nAvailable Experiments:")
for e in mlflow.search_experiments():
    print(f"ID: {e.experiment_id}, Name: {e.name}")