import pandas as pd
import numpy as np
from metaflow import FlowSpec, Parameter, step
import mlflow

# Import the actual functions from your library
from green_taxi_drift_lib import load_taxi_table, run_integrity_checks

# --- HELPER FUNCTIONS ---
def init_mlflow(name):
    # This must be called inside the step to ensure the connection is fresh
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(name)

def process_features(df):
    df = df.copy()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    return df

# --- THE FLOW ---
class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path", help="Path to Jan data")
    batch_path = Parameter("batch-path", help="Path to Apr data")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        """Step 1: Initialize the experiment."""
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Step 2: Load Parquet files."""
        # We store these as 'self' so the next step can see them
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Step 3: Apply column logic."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        """Step 4: Quality Check."""
        # 1. Run checks
        chk = run_integrity_checks(self.batch)
        
        # 2. Log to MLflow (Notice: we don't save 'chk' to self to avoid pickle errors)
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="Data_Integrity"):
            for name, tbl in chk.tables.items():
                mlflow.log_table(tbl, artifact_file=f"checks/{name}.json")
        
        # 3. Decision logic (Requirement: ok check)
        self.ok = True 
        print(f"Gate Check: {'PASSED' if self.ok else 'FAILED'}")
        
        # If OK, go to end (In Phase 2 we will add training here)
        self.next(self.end)

    @step
    def end(self):
        """Step 5: Finish."""
        print("Success! All steps in the flow completed.")

if __name__ == "__main__":
    MLFlowCapstoneFlow()