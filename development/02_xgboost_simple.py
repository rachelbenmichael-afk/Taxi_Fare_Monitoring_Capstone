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

    def objective(self, trial, train_df, test_df, features, target):
    # 1. Define the search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

        # 2. Build and train the model with these trial parameters
        model = self.build_xgboost_model(**params)
        model.fit(train_df[features], train_df[target])

        # 3. Calculate the score for this specific trial
        preds = model.predict(test_df[features])
        rmse = np.sqrt(mean_squared_error(test_df[target], preds))
        
        return rmse
    
    @step
    def train_model(self):
        # 1. Define our features and target based on the Toolbox logic
        features = ['hour', 'day_of_week', 'log_trip_distance']
        target = 'tip_amount'
        
        train_df = self.ref[self.ref['payment_type'] == 1]
        test_df = self.batch[self.batch['payment_type'] == 1]

        # Initialize the Optuna study
        study = optuna.create_study(direction="minimize")
        
        # Run the optimization (20 iterations)
        study.optimize(lambda trial: self.objective(trial, train_df, test_df, features, target), n_trials=20)

        # Get the best parameters and the best score
        self.best_params = study.best_params
        self.rmse = study.best_value
        
        print(f"Optimization complete! Best RMSE: {self.rmse:.4f}")
        print(f"Best Parameters: {self.best_params}")

        
        # Set the active experiment for this step to ensure metrics 
        # and artifacts are logged to 'green_taxi_tip_model' instead of 'Default'.
        init_mlflow(self.model_name)


        # Log the final 'Tuned' model to MLflow
        # CORRECTION: Added 'as run' and moved promotion logic INSIDE this block
        with mlflow.start_run(run_name="Tuned_XGBoost_Candidate") as run:
            mlflow.log_params(self.best_params)
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_metric("rmse", self.rmse)
            
            final_model = self.build_xgboost_model(**self.best_params)
            final_model.fit(train_df[features], train_df[target])
            mlflow.sklearn.log_model(final_model, artifact_path="model")
                     
            print(f"Logging complete! RMSE {self.rmse:.4f} is now in MLflow.")

            # --- PROMOTION LOGIC (Now safely inside the active run) ---
            client = MlflowClient()
            improvement_threshold = 0.01 
            
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                print(f"Candidate ({self.rmse:.4f}) beat Champion ({self.champion_rmse:.4f})!")
                
                # CORRECTION: Using run.info.run_id directly
                result = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", 
                    self.model_name
                )
                
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

    def build_xgboost_model(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0):
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,  # <--- Added this line
                random_state=42,
                objective='reg:squarederror'
            ))
        ])

if __name__ == "__main__":
    MLFlowCapstoneFlow()