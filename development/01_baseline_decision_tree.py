import os
import sys
import xgboost as xgb
import optuna

# --- 2. INTEGRATION POINT (Continue with taxi_flow logic) ---
sys.path.append(os.getcwd())

from metaflow import FlowSpec, Parameter, step
import pandas as pd
import numpy as np
import mlflow
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error 
from mlflow.tracking import MlflowClient # Add this line

# Import your project library
try:
    from green_taxi_drift_lib import load_taxi_table, run_integrity_checks
except ImportError:
    print("Error: Could not find green_taxi_drift_lib.py!")

def init_mlflow(name):
   # This tells Metaflow to send all the metrics to the UI on port 5000
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    
    # This ensures your runs are grouped under the correct project name
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
        print("Checking data integrity...")
        self.chk = run_integrity_checks(self.batch)
        
        # We initialize MLflow for this specific step
        init_mlflow(self.model_name)
        
        with mlflow.start_run(run_name="Data_Integrity"):
            os.makedirs("checks", exist_ok=True)
            for name, tbl in self.chk.tables.items():
                temp_path = f"checks/{name}.csv"
                tbl.to_csv(temp_path, index=False)
                
                # Log to the MLflow UI
                mlflow.log_artifact(temp_path, artifact_path="integrity_reports")
                print(f"Verified, Saved, and Logged: {temp_path}")
            
        self.next(self.load_champion)
    

    @step
    def load_champion(self):
        client = MlflowClient()
        #self.model_name = "green_taxi_tip_model" # Matches your parameter default
        
        try:
            # Try to find the version marked as 'champion'
            champion_version = client.get_model_version_by_alias(self.model_name, "champion")
            # Get the RMSE from that specific run
            run_data = client.get_run(champion_version.run_id).data
            self.champion_rmse = float(run_data.metrics['rmse'])
            print(f"Found Champion! Version: {champion_version.version}, RMSE: {self.champion_rmse}")
        except Exception as e:
            # If no champion exists (Run 1), set a high number so the candidate wins
            print("No champion found in registry. This is the first run.")
            self.champion_rmse = 999.0
            
        self.next(self.train_model)

    
    
    @step
    def train_model(self):
        # 1. Define our features and target based on the Toolbox logic
        # We only train on the 'reference' set (Jan data) for now
        features = ['hour', 'day_of_week', 'log_trip_distance']
        target = 'tip_amount'

            
        # 2. Initialize the model using the blueprint you just added
        self.model = self.build_model()
        
        # 3. Fit the model
        # Train on Jan (reference)
        # We filter for credit card only as suggested by the data documentation
        train_df = self.ref[self.ref['payment_type'] == 1]
        self.model.fit(train_df[features], train_df[target])
        
        # --- NEW: Evaluation (Step E) ---
        # Evaluate on Apr (batch)
        test_df = self.batch[self.batch['payment_type'] == 1]
        preds = self.model.predict(test_df[features])
        
        # Calculate RMSE
        self.rmse = np.sqrt(mean_squared_error(test_df[target], preds))
        print(f"Model training complete! Batch RMSE: {self.rmse:.4f}")

        # Set the active experiment for this step to ensure metrics 
        # and artifacts are logged to 'green_taxi_tip_model' instead of 'Default'.
        init_mlflow(self.model_name)

        # Start an MLflow run to record this training session
        with mlflow.start_run(run_name="Baseline_DecisionTree") as run:
            # 1. Log the "Settings" (Parameters)
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("max_depth", 8)
                           
            # 2. Log the "Result" (Metric)
            mlflow.log_metric("rmse", self.rmse)
            
            # 3. Log the "Brain" (The actual Model)
            mlflow.sklearn.log_model(self.model, artifact_path="model")
            
            print(f"Logging complete! RMSE {self.rmse:.4f} is now in MLflow.")

            # --- PROMOTION LOGIC (Now inside the 'with' block) ---
            client = MlflowClient()
            improvement_threshold = 0.01 # 1% improvement
            
            # Check if candidate is at least 1% better than current champion
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                print(f"Candidate ({self.rmse:.4f}) beat Champion ({self.champion_rmse:.4f})!")
                
                # Register the new model version using the current active run
                result = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", 
                    self.model_name
                )
                
                # Move the '@champion' alias to this new version
                client.set_registered_model_alias(self.model_name, "champion", result.version)
                print(f"Model version {result.version} is now the @champion.")
            else:
                print("Candidate did not improve enough. Champion remains the same.")

        self.next(self.end)
                        
                                    
            

    @step
    def end(self):
        print("Success! Phase 1 is complete.")
    
    def build_model(self, random_state: int = 0, max_depth: int = 8, min_samples_leaf: int = 200) -> Pipeline:
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