import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import nannyml as nml
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
    Initializes tracking for the champion promotion phase.
    Ensures artifacts are saved to the local workspace for auditable decisions.
    """
    abs_tracking_uri = "file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns"
    mlflow.set_tracking_uri(abs_tracking_uri)
    mlflow.set_experiment(name)

def process_features(df):
    """
    Design Doc Step C: Feature Engineering
    Implements advanced features including duration, speed, and location embeddings.
    Used consistently for training, evaluation, and monitoring.
    """
    df = df.copy()
    
    # 1. Time Features
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    
    # 2. Mathematical Transforms
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    
    # 3. Temporal Features (Duration in minutes)
    duration = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['trip_duration'] = duration.clip(lower=1, upper=120)
    
    # 4. Behavioral Features (Average Speed)
    df['avg_speed'] = df['trip_distance'] / (df['trip_duration'])
    
    # 5. Spatial Features (Location IDs)
    df['pickup_zip'] = df['PULocationID'].astype(int)
    df['dropoff_zip'] = df['DOLocationID'].astype(int)
    
    return df

class TaxiMonitoringFlow(FlowSpec):
    """
    Phase 2: Automated Retraining and Promotion Logic.
    Integrates NannyML DLE to estimate performance and decide on retraining.
    """
    reference_path = Parameter("reference-path", help="April data (Stable Baseline)")
    batch_path = Parameter("batch-path", help="August data (New Production Batch)")
    model_name = Parameter("model-name", default="green_taxi_tip_model_final")

    @step
    def start(self):
        """Initialize experiment and prepare for data ingestion."""
        init_mlflow(self.model_name)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Design Doc Step A: Load reference and current production batches."""
        self.ref = load_taxi_table(self.reference_path)
        self.batch = load_taxi_table(self.batch_path)
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """Design Doc Step C: Apply advanced feature engineering to both datasets."""
        self.ref = process_features(self.ref)
        self.batch = process_features(self.batch)
        self.next(self.integrity_gate)
    
    @step
    def integrity_gate(self):
        """
        Design Doc Step B: Two-Layer Integrity Gate
        Layer 1: Hard Rules (via lib)
        Layer 2: NannyML Missingness Spike (Soft Gate)
        """
        self.chk = run_integrity_checks(self.batch)
        init_mlflow(self.model_name)
        
        with mlflow.start_run(run_name="Data_Integrity"):
            os.makedirs("checks", exist_ok=True)
            for name, tbl in self.chk.tables.items():
                temp_path = f"checks/{name}.csv"
                tbl.to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path, artifact_path="integrity_reports")

        # NannyML Soft Gate Check
        try:
            feature_cols = ['log_trip_distance', 'trip_duration', 'avg_speed']
            calc = nml.MissingValuesCalculator(column_names=feature_cols)
            calc.fit(self.ref)
            results = calc.calculate(self.batch)
            alerts = results.filter(period='analysis').to_df().iloc[-1].filter(like='alert')
            if alerts.any():
                print("⚠️ NannyML ALERT: Missingness spike detected!")
        except Exception as e:
            print(f"Soft gate check failed: {e}")
                         
        self.next(self.load_champion)

    @step
    def load_champion(self):
        """Design Doc Step D: Retrieve active champion for performance benchmarking."""
        init_mlflow(self.model_name)
        client = MlflowClient()
        try:
            champion_version = client.get_model_version_by_alias(self.model_name, "champion")
            run_data = client.get_run(champion_version.run_id).data
            self.champion_rmse = float(run_data.metrics['rmse'])
            print(f"Champion Active: Version {champion_version.version}")
        except Exception:
            print("No champion found. Bootstrapping first model.")
            self.champion_rmse = 999.0 

        self.next(self.model_gate)

    @step
    def model_gate(self):
        """
        Design Doc Step E: Performance Gate
        Uses NannyML DLE to estimate Champion RMSE on the new batch without targets.
        """
        init_mlflow(self.model_name)
        client = MlflowClient()
        self.retrain_needed = False 

        try:
            champ_version = client.get_model_version_by_alias(self.model_name, "champion")
            from mlflow.sklearn import load_model
            model = load_model(f"models:/{self.model_name}@champion")
            
            features = ['hour', 'day_of_week', 'log_trip_distance', 'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip']
            self.ref['prediction'] = model.predict(self.ref[features])
            self.batch['prediction'] = model.predict(self.batch[features])
            
            # DLE Performance Estimation
            estimator = nml.DLE(
                feature_column_names=features, y_pred='prediction', y_true='tip_amount',
                metrics=['rmse'], chunk_size=5000 
            )
            estimator.fit(self.ref)
            results = estimator.estimate(self.batch)
            
            self.estimated_rmse = results.to_df().iloc[-1][('rmse', 'value')]
            
            # Decision Rule: Retrain if estimated RMSE exceeds baseline by 5%
            drift_threshold = 1.05 
            self.retrain_needed = self.estimated_rmse > (self.champion_rmse * drift_threshold)

        except Exception:
            print("Forcing retrain to establish first champion.")
            self.retrain_needed = True

        self.route = "retrain" if self.retrain_needed else "skip"
        self.next({"retrain": self.train_model, "skip": self.end}, condition="route")
    
    def objective(self, trial, train_df, test_df, features, target):
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
        Design Doc Step F: Conditional Retraining
        Trains a new XGBoost candidate if the Performance Gate detects degradation.
        """
        features = ['hour', 'day_of_week', 'log_trip_distance', 'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip']
        target = 'tip_amount'
        train_df = self.ref[self.ref['payment_type'] == 1]
        test_df = self.batch[self.batch['payment_type'] == 1]

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_df, test_df, features, target), n_trials=20)

        self.best_params = study.best_params
        self.rmse = study.best_value
        init_mlflow(self.model_name)

        with mlflow.start_run(run_name="Tuned_XGBoost_Candidate_Plus_FE") as run:
            mlflow.log_params(self.best_params)
            mlflow.log_metric("rmse", self.rmse)
            
            final_model = self.build_xgboost_model(**self.best_params)
            final_model.fit(train_df[features], train_df[target])
            mlflow.sklearn.log_model(final_model, artifact_path="model")

            # Design Doc Step G: Candidate Acceptance (Promotion)
            client = MlflowClient()
            improvement_threshold = 0.00 
            
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                result = mlflow.register_model(f"runs:/{run.info.run_id}/model", self.model_name)
                client.set_registered_model_alias(self.model_name, "champion", result.version)
                print(f"Version {result.version} promoted to @champion.")

        self.next(self.end)

    @step
    def end(self):
        """Report flow completion status."""
        status = "RE-TRAINED" if getattr(self, 'retrain_needed', False) else "MONITORED"
        print (f"Flow complete. Status: {status}")

    def build_xgboost_model(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0):
        """Pipeline factory for XGBoost regressor."""
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, subsample=subsample,
                random_state=42, objective='reg:squarederror'
            ))
        ])

if __name__ == "__main__":
    TaxiMonitoringFlow()