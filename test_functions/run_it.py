import pandas as pd
import numpy as np
import mlflow
import os
import sys

# Ensure local folder is visible for your library
sys.path.append(os.getcwd())

from green_taxi_drift_lib import load_taxi_table, run_integrity_checks

# Configuration
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "green_taxi_tip_model"
REF_PATH = "TLC_data/green_tripdata_2020-01.parquet"
BATCH_PATH = "TLC_data/green_tripdata_2020-04.parquet"

def process_features(df):
    """ Your required feature engineering logic """
    df = df.copy()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    # Clipping and log transform for distance
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    return df

def run_project():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("--- Starting Phase 1 ---")
    
    print("Step 1: Loading data...")
    ref = load_taxi_table(REF_PATH)
    batch = load_taxi_table(BATCH_PATH)
    
    print("Step 2: Running Feature Engineering...")
    ref = process_features(ref)
    batch = process_features(batch)
    
    print("Step 3: Running Integrity Gate...")
    chk = run_integrity_checks(batch)
    
    # Log to MLflow
    with mlflow.start_run(run_name="Manual_Data_Integrity"):
        print("Logging results to MLflow...")
        os.makedirs("checks", exist_ok=True)
        for name, tbl in chk.tables.items():
            temp_path = f"checks/{name}.csv"
            tbl.to_csv(temp_path, index=False)
            mlflow.log_artifact(temp_path, artifact_path="integrity_reports")
            
    print("\nSuccess! Phase 1 complete.")
    print("You can now see the 'Manual_Data_Integrity' run in your MLflow UI.")

if __name__ == "__main__":
    run_project()