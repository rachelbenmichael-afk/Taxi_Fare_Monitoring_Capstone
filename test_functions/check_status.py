import mlflow
import os
from mlflow.tracking import MlflowClient

# Set the path to your local mlruns folder
tracking_path = "file://" + os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(tracking_path)

client = MlflowClient()
model_name = "green_taxi_tip_model"

try:
    # Look for the version tagged as 'champion'
    champ = client.get_model_version_by_alias(model_name, "champion")
    print(f"\n✅ CURRENT CHAMPION FOUND")
    print(f"--------------------------")
    print(f"Model Name:      {model_name}")
    print(f"Champion Version: {champ.version}")
    print(f"Source Run ID:    {champ.run_id}")
    print(f"--------------------------\n")
except Exception as e:
    print(f"\n⚠️  No champion found yet.")
    print(f"Reason: {e}\n")