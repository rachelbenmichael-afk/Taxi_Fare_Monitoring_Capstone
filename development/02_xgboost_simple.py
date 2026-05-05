import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
from mlflow.tracking import MlflowClient
from metaflow import FlowSpec, Parameter, step
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
    Initializes MLflow tracking settings.
    Ensures all runs are grouped under the correct experiment and persisted locally.
    """
    abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
    mlflow.set_tracking_uri(abs_tracking_uri)
    mlflow.set_experiment(name)

def process_features(df):
    """
    Design Doc Step C: Feature Engineering
    Applies the standardized feature set: time features and log-transformed distance.
    """
    df = df.copy()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    return df

class MLFlowCapstoneFlow(FlowSpec):
    """
    Phase 1 (Advanced): XGBoost Optimization
    Introduces hyperparameter tuning via Optuna and compares against the Phase 1 baseline.
    """
    reference_path = Parameter("reference-path", help="Path to January data")
    batch_path = Parameter("batch-path", help="Path to April data")
    model_name = Parameter("model-name", default="green_taxi_tip_model_final")

    @step
    def start(self):
        """Entry point: Initialize tracking and proceed to data loading."""
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Design Doc Step A: Load datasets for reference and evaluation."""
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Design Doc Step C: Generate model-ready features for both slices."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        """
        Design Doc Step B: Integrity Gate
        Logs raw data quality metrics to MLflow before proceeding to training.
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
        Design Doc Step D: Load Current Champion
        Retrieves the performance of the current champion to set a benchmark for the candidate.
        """
        client = MlflowClient()
        try:
            champion_version = client.get_model_version_by_alias(self.model_name, "champion")
            run_data = client.get_run(champion_version.run_id).data
            self.champion_rmse = float(run_data.metrics['rmse'])
            print(f"Found Champion! Version: {champion_version.version}, RMSE: {self.champion_rmse}")
        except Exception:
            print("No champion found. Setting high baseline for bootstrap.")
            self.champion_rmse = 999.0
            
        self.next(self.train_model)

    def objective(self, trial, train_df, test_df, features, target):
        """Optuna objective function for hyperparameter search."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = self.build_xgboost_model(**params)
        model.fit(train_df[features], train_df[target])
        preds = model.predict(test_df[features])
        return np.sqrt(mean_squared_error(test_df[target], preds))
    
    @step
    def train_model(self):
        """
        Design Doc Step F: Retrain (with Optimization)
        Runs Optuna study to find best XGBoost parameters and trains final candidate.
        """
        features = ['hour', 'day_of_week', 'log_trip_distance']
        target = 'tip_amount'
        
        train_df = self.ref[self.ref['payment_type'] == 1]
        test_df = self.batch[self.batch['payment_type'] == 1]

        # Optimization study
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_df, test_df, features, target), n_trials=20)

        self.best_params = study.best_params
        self.rmse = study.best_value
        
        init_mlflow(self.model_name)

        # Log the Tuned XGBoost Candidate
        with mlflow.start_run(run_name="Tuned_XGBoost_Candidate") as run:
            mlflow.log_params(self.best_params)
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_metric("rmse", self.rmse)
            
            final_model = self.build_xgboost_model(**self.best_params)
            final_model.fit(train_df[features], train_df[target])
            mlflow.sklearn.log_model(final_model, artifact_path="model")

            # Step G: Candidate Acceptance & Promotion
            client = MlflowClient()
            improvement_threshold = 0.01 
            
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                print(f"Candidate ({self.rmse:.4f}) beat Champion ({self.champion_rmse:.4f})!")
                result = mlflow.register_model(f"runs:/{run.info.run_id}/model", self.model_name)
                client.set_registered_model_alias(self.model_name, "champion", result.version)
                print(f"Model version {result.version} is now the @champion.")
            else:
                print("Candidate did not improve enough.")

        self.next(self.end)
                                                      
    @step
    def end(self):
        """Finalize the XGBoost optimization flow."""
        print("Success! XGBoost Phase 1 optimization complete.")

    def build_xgboost_model(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0):
        """Pipeline helper for the XGBoost regressor."""
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42,
                objective='reg:squarederror'
            ))
        ])

if __name__ == "__main__":
    MLFlowCapstoneFlow()