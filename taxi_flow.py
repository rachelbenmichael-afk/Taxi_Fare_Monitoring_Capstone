import os
import sys

# --- 1. THE WINDOWS BYPASS (Start copying here) ---
if sys.platform == "win32":
    # Manually inject missing constants
    if not hasattr(os, 'O_NONBLOCK'):
        os.O_NONBLOCK = 0
    
    # Force Metaflow to use 'local' mode to avoid the Polling engine
    os.environ['METAFLOW_METADATA'] = 'local'
    os.environ['METAFLOW_RUN_SIDECAR'] = '0'
    os.environ['METAFLOW_RESCUE_SIDECAR'] = '0'

    # Mock fcntl for Windows
    try:
        import fcntl
    except ImportError:
        class DummyFcntl:
            F_SETFL = 0; F_GETFL = 0
            def fcntl(self, fd, op, arg=0): return 0
            def ioctl(self, fd, op, arg=0): return 0
        sys.modules['fcntl'] = DummyFcntl()

# --- 2. INTEGRATION POINT (Continue with taxi_flow logic) ---
sys.path.append(os.getcwd())

from metaflow import FlowSpec, Parameter, step
import pandas as pd
import numpy as np
import mlflow

# Import your project library
try:
    from green_taxi_drift_lib import load_taxi_table, run_integrity_checks
except ImportError:
    print("Error: Could not find green_taxi_drift_lib.py!")

def init_mlflow(name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(name)

def process_features(df):
    df = df.copy()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    return df

class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path", help="Path to Jan data")
    batch_path = Parameter("batch-path", help="Path to Apr data")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        chk = run_integrity_checks(self.batch)
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="Data_Integrity"):
            os.makedirs("checks", exist_ok=True)
            for name, tbl in chk.tables.items():
                temp_path = f"checks/{name}.csv"
                tbl.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path, artifact_path="integrity_reports")
        self.next(self.end)

    @step
    def end(self):
        print("Success! Phase 1 is complete.")

if __name__ == "__main__":
    MLFlowCapstoneFlow()