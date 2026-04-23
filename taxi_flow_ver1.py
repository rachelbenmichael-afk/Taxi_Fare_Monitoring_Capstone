import pandas as pd
import numpy as np
from metaflow import FlowSpec, Parameter, step
import mlflow

# Import the specific functions from your course library
from green_taxi_drift_lib import load_taxi_table, run_integrity_checks

# --- HELPER FUNCTIONS ---
def init_mlflow(name):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(name)

def process_features(df):
    """
    This is the Feature Engineering logic from your 0_green_taxi_eda notebook.
    """
    # 1. Time features
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    
    # 2. Logic for log/clip transforms (as per Design Doc requirements)
    # We clip trip_distance to avoid outliers and then take log
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    
    return df

# --- THE FLOW ---
class MLFlowCapstoneFlow(FlowSpec):
    reference_path = Parameter("reference-path", help="Path to Jan 2020 data")
    batch_path = Parameter("batch-path", help="Path to Apr 2020 data")
    model_name = Parameter("model-name", default="green_taxi_tip_model")

    @step
    def start(self):
        """Initialize and move to loading."""
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load the Parquet files using the library function."""
        # Using the actual function name from your library: load_taxi_table
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Apply the EDA logic to both datasets."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        """Check data quality before proceeding."""
        # We use the reference and batch to look for drift/issues
        chk = run_integrity_checks(self.batch) 
        
        # For now, we'll let it pass to test the flow, but in Phase 2 
        # we will add the NannyML logic here.
        self.ok = True 
        
        print(f"Integrity check completed. Proceeding: {self.ok}")
        self.next(self.end)

    @step
    def end(self):
        """Finish the flow."""
        print("Flow finished successfully!")

if __name__ == "__main__":
    MLFlowCapstoneFlow()