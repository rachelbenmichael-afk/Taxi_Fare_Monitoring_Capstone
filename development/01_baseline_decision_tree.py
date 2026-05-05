import os
import sys
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# --- Project Library Integration ---
sys.path.append(os.getcwd())
try:
    from green_taxi_drift_lib import load_taxi_table, run_integrity_checks
except ImportError:
    print("Error: Could not find green_taxi_drift_lib.py!")

def init_mlflow(name):
    """
    Initializes MLflow tracking settings for GitHub Codespaces environment.
    Anchors all artifacts to the local workspace and groups runs by experiment name.
    """
    abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
    mlflow.set_tracking_uri(abs_tracking_uri)
    mlflow.set_experiment(name)

def process_features(df):
    """
    Design Doc Step C: Feature Engineering
    Transforms raw columns into model-ready features, including time-based 
    attributes and log transformations for trip distance.
    """
    df = df.copy()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    return df

class MLFlowCapstoneFlow(FlowSpec):
    """
    Phase 1: Baseline Orchestration
    Establishes the initial champion model using a Decision Tree regressor.
    """
    reference_path = Parameter("reference-path", help="Path to January training data")
    batch_path = Parameter("batch-path", help="Path to April evaluation data")
    model_name = Parameter("model-name", default="green_taxi_tip_model_final")

    @step
    def start(self):
        """Initialize experiment logging and start the flow."""
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Design Doc Step A: Load reference and batch datasets."""
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Design Doc Step C: Apply consistent transformations to both data slices."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        """
        Design Doc Step B: Integrity Gate
        Runs hard schema checks and logs results to MLflow.
        """
        print("Checking data integrity...")
        self.chk = run_integrity_checks(self.batch)
        
        init_mlflow(self.model_name)
        with mlflow.start_run(run_name="Data_Integrity"):
            os.makedirs("checks", exist_ok=True)
            for name, tbl in self.chk.tables.items():
                temp_path = f"checks/{name}.csv"
                tbl.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path, artifact_path="integrity_reports")
            
        self.next(self.load_champion)

    @step
    def load_champion(self):
        """
        Design Doc Step D: Load Champion Model
        Attempts to retrieve the current @champion from the MLflow Model Registry.
        """
        client = MlflowClient()
        try:
            champion_version = client.get_model_version_by_alias(self.model_name, "champion")
            run_data = client.get_run(champion_version.run_id).data
            self.champion_rmse = float(run_data.metrics['rmse'])
            print(f"Found Champion! Version: {champion_version.version}, RMSE: {self.champion_rmse}")
        except Exception:
            print("No champion found in registry. This is the bootstrap run.")
            self.champion_rmse = 999.0
            
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Design Doc Step F & G: Training and Candidate Acceptance
        Trains a baseline model and promotes it if it meets the improvement threshold.
        """
        features = ['hour', 'day_of_week', 'log_trip_distance']
        target = 'tip_amount'

        # Build and fit the baseline model
        self.model = self.build_model()
        train_df = self.ref[self.ref['payment_type'] == 1]
        self.model.fit(train_df[features], train_df[target])
        
        # Step E: Evaluate performance on the Batch (April)
        test_df = self.batch[self.batch['payment_type'] == 1]
        preds = self.model.predict(test_df[features])
        self.rmse = np.sqrt(mean_squared_error(test_df[target], preds))

        init_mlflow(self.model_name)

        # Log training run to MLflow
        with mlflow.start_run(run_name="Baseline_DecisionTree") as run:
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("max_depth", 8)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.sklearn.log_model(self.model, artifact_path="model")
            
            # Step G: Promotion logic (Candidate Acceptance)
            client = MlflowClient()
            improvement_threshold = 0.01 
            
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                print(f"Candidate ({self.rmse:.4f}) beat Champion ({self.champion_rmse:.4f})!")
                result = mlflow.register_model(f"runs:/{run.info.run_id}/model", self.model_name)
                client.set_registered_model_alias(self.model_name, "champion", result.version)
                print(f"Model version {result.version} is now the @champion.")
            else:
                print("Candidate did not improve enough. Champion remains the same.")

        self.next(self.end)

    @step
    def end(self):
        """Log successful completion of the Phase 1 flow."""
        print("Success! Phase 1 baseline established.")
    
    def build_model(self, random_state: int = 0, max_depth: int = 8, min_samples_leaf: int = 200) -> Pipeline:
        """Helper to construct the sklearn pipeline with imputer and regressor."""
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("tree", DecisionTreeRegressor(
                random_state=random_state,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )),
        ])

if __name__ == "__main__":
    MLFlowCapstoneFlow()