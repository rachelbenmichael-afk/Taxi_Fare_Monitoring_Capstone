import os
import sys
import pandas as pd
import numpy as np
import mlflow
import nannyml as nml
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step

# --- 1. DEFINE INIT_MLFLOW DIRECTLY HERE ---
def init_mlflow(name):
    # This tells Metaflow to send all the metrics to the UI on port 5000 (old)
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # This anchors all your scripts to the same physical location
    abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
    mlflow.set_tracking_uri(abs_tracking_uri)

    mlflow.set_experiment(name)

# --- 2. THE IMPORT HACK FOR YOUR FEATURES ---
import importlib
sys.path.append(os.getcwd())
try:
    training_module = importlib.import_module("03_xgboost_optimized_champion")
    process_features = training_module.process_features
    print("Successfully imported process_features logic!")
except ImportError as e:
    print(f"Error: Could not find training file logic: {e}")

# --- 3. START THE CLASS ---

class TaxiMonitoringFlow(FlowSpec):
    model_name = Parameter("model-name", default="green_taxi_tip_model")
    reference_path = Parameter("reference-path", help="Jan data")
    batch_path = Parameter("batch-path", help="Current data batch")

    @step
    def start(self):
        init_mlflow(self.model_name)
        self.ref = pd.read_parquet(self.reference_path)
        self.batch = pd.read_parquet(self.batch_path)
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.run_monitoring)

    
    @step
    def run_monitoring(self):
        # 1. FIX: Use the SAME physical path as your other scripts
        abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
        os.environ["MLFLOW_TRACKING_URI"] = abs_tracking_uri
        mlflow.set_tracking_uri(abs_tracking_uri)
        
        client = MlflowClient()
        
        # 2. Load the Champion model 
        try:
            # We use the tracking_uri we just set above
            champion_ver = client.get_model_version_by_alias(self.model_name, "champion")
            model_uri = f"models:/{self.model_name}@champion"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"Successfully loaded champion version {champion_ver.version}")
        except Exception as e:
            print(f"Failed to find model '{self.model_name}' with alias '@champion'")
            raise e
        
        # Define features used in training
        features = ['hour', 'day_of_week', 'log_trip_distance', 'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip']
        
        # 3. Generate predictions needed for drift analysis
        self.ref['prediction'] = model.predict(self.ref[features])
        self.batch['prediction'] = model.predict(self.batch[features])

        # 4. Start an MLflow Run specifically for Monitoring
        with mlflow.start_run(run_name="Monitoring_Deep_Dive"):
            os.makedirs("monitoring_plots", exist_ok=True)
            
            # --- UNIVARIATE DRIFT ---
            calc = nml.UnivariateDriftCalculator(
                column_names=features,
                continuous_methods=['kolmogorov_smirnov'],
                chunk_size=5000
            )
            calc.fit(self.ref)
            results = calc.calculate(self.batch)
            
            # Save and Log Plot
            fig = results.filter(column_names=features).plot()
            fig.write_html("monitoring_plots/univariate_drift.html")
            
            # --- PERFORMANCE ESTIMATION (DLE) ---
            estimator = nml.DLE(
                feature_column_names=features,
                y_pred='prediction',
                y_true='tip_amount',
                metrics=['rmse'],
                chunk_size=5000
            )
            estimator.fit(self.ref)
            est_results = estimator.estimate(self.batch)
            
            # Save and Log Plot
            fig_perf = est_results.plot()
            fig_perf.write_html("monitoring_plots/performance_estimation.html")

            # Upload the whole folder to MLflow
            mlflow.log_artifacts("monitoring_plots", artifact_path="drift_analysis")
            print("Monitoring artifacts successfully uploaded to MLflow.")

        self.next(self.end)
    @step
    def end(self):
        print("Monitoring Flow Complete.")

if __name__ == "__main__":
    TaxiMonitoringFlow()