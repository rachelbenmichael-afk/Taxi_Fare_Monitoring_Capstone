import os
import sys
import pandas as pd
import numpy as np
import mlflow
import nannyml as nml
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step

# --- Project Library Integration ---
def init_mlflow(name):
    """
    Anchors monitoring results to the central MLflow repository.
    Ensures that drift reports and performance estimations are grouped with the main experiment.
    """
    abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
    mlflow.set_tracking_uri(abs_tracking_uri)
    mlflow.set_experiment(name)

# --- Feature Engineering Import ---
import importlib
sys.path.append(os.getcwd())
try:
    # Dynamically import the feature logic from the champion script to ensure consistency
    training_module = importlib.import_module("03_xgboost_optimized_champion")
    process_features = training_module.process_features
    print("Successfully imported process_features logic!")
except ImportError as e:
    print(f"Error: Could not find training file logic: {e}")

class TaxiMonitoringFlow(FlowSpec):
    """
    Phase 3: Deep-Dive Monitoring & Reporting.
    This flow generates detailed NannyML reports for Univariate Drift and Performance Estimation.
    It is used to visualize data shifts between the January baseline and current production batches.
    """
    model_name = Parameter("model-name", default="green_taxi_tip_model_final")
    reference_path = Parameter("reference-path", help="January baseline data")
    batch_path = Parameter("batch-path", help="Current production data batch")

    @step
    def start(self):
        """Initialize tracking and load raw data slices."""
        init_mlflow(self.model_name)
        self.ref = pd.read_parquet(self.reference_path)
        self.batch = pd.read_parquet(self.batch_path)
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Design Doc Step C: Apply standardized feature engineering to ensure valid comparison."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.run_monitoring)
        # --- TEMPORARY CRASH FOR DEMO ---
        # This is aligned with self.next
        # raise Exception("Simulated Pipeline Failure for Demonstration")
    
    @step
    def run_monitoring(self):
        
        """
        Main Monitoring Step.
        1. Loads the current Champion model.
        2. Calculates Univariate Drift (KS Test).
        3. Estimates RMSE using DLE.
        4. Uploads interactive HTML reports as MLflow artifacts.
        """
        # Ensure tracking URI is set for this step's environment
        abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
        os.environ["MLFLOW_TRACKING_URI"] = abs_tracking_uri
        mlflow.set_tracking_uri(abs_tracking_uri)
        
        client = MlflowClient()
        
        # Load the current Champion model from the Registry
        try:
            champion_ver = client.get_model_version_by_alias(self.model_name, "champion")
            model_uri = f"models:/{self.model_name}@champion"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded champion version {champion_ver.version}")
        except Exception as e:
            print(f"Failed to find @champion for monitoring: {e}")
            raise e
        
        # Define the specific features to monitor for drift
        features = ['hour', 'day_of_week', 'log_trip_distance', 'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip']
        
        # Generate predictions required for Drift and Estimation analysis
        self.ref['prediction'] = model.predict(self.ref[features])
        self.batch['prediction'] = model.predict(self.batch[features])

        # Execute detailed NannyML analysis
        init_mlflow(self.model_name) # Ensure it uses the correct experiment
        
        with mlflow.start_run(run_name="Monitoring_Deep_Dive"):
            # Adding tags
            mlflow.set_tag("type", "monitoring_output")
            mlflow.set_tag("batch", os.path.basename(self.batch_path)) # Dynamic file name
            
            os.makedirs("monitoring_plots", exist_ok=True)
            
            # --- UNIVARIATE DRIFT ANALYSIS ---
            # Compares distribution shapes between Reference and Analysis periods.
            calc = nml.UnivariateDriftCalculator(
                column_names=features,
                continuous_methods=['kolmogorov_smirnov'],
                chunk_size=5000
            )
            calc.fit(self.ref)
            results = calc.calculate(self.batch)
            
            # Save Drift Report
            fig = results.filter(column_names=features).plot()
            fig.write_html("monitoring_plots/univariate_drift.html")
            
            # --- PERFORMANCE ESTIMATION (DLE) ---
            # Estimates actual error (RMSE) on unlabeled production data.
            estimator = nml.DLE(
                feature_column_names=features,
                y_pred='prediction',
                y_true='tip_amount',
                metrics=['rmse'],
                chunk_size=5000
            )
            estimator.fit(self.ref)
            est_results = estimator.estimate(self.batch)
            
            # Save Performance Estimation Report
            fig_perf = est_results.plot()
            fig_perf.write_html("monitoring_plots/performance_estimation.html")

            # Upload interactive reports to MLflow UI
            mlflow.log_artifacts("monitoring_plots", artifact_path="drift_analysis")
            print("Monitoring reports uploaded as artifacts.")

        self.next(self.end)

    @step
    def end(self):
        """Finalize the deep-dive monitoring flow."""
        print("Monitoring Flow Complete. Reports available in MLflow Artifacts.")

if __name__ == "__main__":
    TaxiMonitoringFlow()