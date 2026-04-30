# 1. Imports
import os
import sys
import xgboost as xgb
import optuna
import nannyml as nml

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

# 2. Project Library Imports
try:
    from green_taxi_drift_lib import load_taxi_table, run_integrity_checks
except ImportError:
    print("Error: Could not find green_taxi_drift_lib.py!")

# 3. Helper Functions

def init_mlflow(name):
   # mlflow.set_tracking_uri("http://localhost:5000") i have skipped on this as we are using mlflow in local mode and it will create a mlruns directory in the current working directory to store the runs and artifacts.
   # mlflow.set_experiment(name)
   # We stay in local mode (no set_tracking_uri needed)
   # This creates a solid path to the 'mlruns' folder in your project
    tracking_path = "file://" + os.path.join(os.getcwd(), "mlruns")
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment(name)

def process_features(df):
    df = df.copy()
    
    # 1. Time Features
    df['hour'] = df['lpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['lpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['lpep_pickup_datetime'].dt.month
    
    # 2. Advanced Mathematical Transforms
    df['log_trip_distance'] = np.log1p(df['trip_distance'].clip(lower=0, upper=100))
    
    # 3. NEW: Trip Duration (in minutes)
    duration = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['trip_duration'] = duration.clip(lower=1, upper=120) # Clip to remove outliers/errors
    
    # 4. NEW: Average Speed (miles per minute)
    df['avg_speed'] = df['trip_distance'] / (df['trip_duration'])
    
    # 5. NEW: Location Features (Cast to int to ensure stability)
    df['pickup_zip'] = df['PULocationID'].astype(int)
    df['dropoff_zip'] = df['DOLocationID'].astype(int)
    
    return df

class TaxiMonitoringFlow(FlowSpec):
    # UPDATE: Changing the help text to match the Phase 2 scenario
    reference_path = Parameter(
        "reference-path", 
        help="Path to April data (your stable baseline)",
        default="green_tripdata_2020-04.parquet" # Optional: set a default
    )
    batch_path = Parameter(
        "batch-path", 
        help="Path to August data (the new batch to monitor)",
        default="green_tripdata_2020-08.parquet" # Optional: set a default
    )
    model_name = Parameter(
        "model-name", 
        default="green_taxi_tip_model"
    )


    @step
    def start(self):
        init_mlflow(self.model_name)
        self.next(self.load_data)

    # ... (Keep load_data, feature_engineering, integrity_gate)
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
        print("Running Layer 1 (Hard Checks) & Layer 2 (NannyML Soft Checks)...")
        
        # 1. Layer 1: Capture the single object returned by your library
        self.chk = run_integrity_checks(self.batch)
        
        # Save every individual table from the checks
        os.makedirs("checks", exist_ok=True)
        for name, tbl in self.chk.tables.items():
            temp_path = f"checks/{name}.csv"
            tbl.to_csv(temp_path, index=False)
            print(f"Verified and Saved: {temp_path}")

        # 2. Layer 2: NannyML Missing Values Spike Check (Soft Gate)
        # This compares April (ref) to August (batch)
        try:
            feature_cols = ['log_trip_distance', 'trip_duration', 'avg_speed']
            calc = nml.MissingValuesCalculator(column_names=feature_cols)
            
            # Fit on April, calculate on August
            calc.fit(self.ref)
            results = calc.calculate(self.batch)
            
            # Check for alerts in the latest data chunk
            alerts = results.filter(period='analysis').to_df().iloc[-1].filter(like='alert')
            if alerts.any():
                print("⚠️  NannyML ALERT: Significant missingness spike detected in August data!")
            else:
                print("✅ NannyML: No missingness spikes detected.")
        except Exception as e:
            print(f"Could not run NannyML Layer 2 checks: {e}")
                         
        # We always proceed to load_champion because Layer 1 didn't crash 
        # and Layer 2 is a 'soft' warning gate.
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

        self.next(self.model_gate) # CHANGED: Go to the gate first

    @step
    def model_gate(self):
        """
        Step E: Performance Gate using NannyML
        Estimates the champion's performance on the new batch.
        """
        print("Evaluating Champion performance on new batch...")
        
        # 1. Prepare data for NannyML
        features = ['hour', 'day_of_week', 'log_trip_distance', 'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip']
        target = 'tip_amount'
        
        # NannyML needs a reference period (April) to 'learn' the error patterns
        # and an analysis period (August) to estimate the current error.
        
        # 2. Initialize NannyML DLE (Direct Loss Estimator)
        estimator = nml.DLE(
            feature_column_names=features,
            y_pred='prediction', # We'll need to add predictions to our dataframes
            y_true=target,
            metrics=['rmse'],
            chunk_size=5000 
        )
        
        # We must add the champion's predictions to both dataframes
        # This simulates running the champion model on the data
        from mlflow.sklearn import load_model
        client = MlflowClient()
        champ_version = client.get_model_version_by_alias(self.model_name, "champion")
        model = load_model(f"models:/{self.model_name}@champion")
        
        self.ref['prediction'] = model.predict(self.ref[features])
        self.batch['prediction'] = model.predict(self.batch[features])
        
        # 3. Fit on reference (April) and estimate on analysis (August)
        estimator.fit(self.ref)
        results = estimator.estimate(self.batch)
        
        # Get the estimated RMSE value
        self.estimated_rmse = results.to_df().iloc[-1][('rmse', 'value')]
        print(f"Champion Baseline RMSE: {self.champion_rmse:.4f}")
        print(f"Estimated RMSE on New Batch: {self.estimated_rmse:.4f}")
        
        # 4. Decision Logic: Retrain if estimated RMSE is > 10% worse than baseline
        drift_threshold = 1.05 
        self.retrain_needed = self.estimated_rmse > (self.champion_rmse * drift_threshold)
        
        if self.retrain_needed:
            print("ALERT: Performance degradation detected. Triggering retraining.")
        else:
            print("Performance is stable. No retraining needed today.")

        # Step F: Conditional transition
        #self.next(self.train_model if self.retrain_needed else self.end)
        # Determine the route based on the NannyML result
        if self.retrain_needed:
            self.route = "retrain"
        else:
            self.route = "skip"

        # This dictionary tells Metaflow: 
        # "If self.route is 'retrain', go to self.train_model. 
        #  If it's 'skip', go to self.end."
        self.next(
            {"retrain": self.train_model, "skip": self.end}, 
            condition="route"
        )
    
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
        """
        Step F: Retrain (Conditional)
        Only runs if self.retrain_needed was True.
        """
        # Update this list to include your new features!
        features = [
            'hour', 'day_of_week', 'log_trip_distance', 
            'trip_duration', 'avg_speed', 'pickup_zip', 'dropoff_zip'
        ]
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

        # Log the final 'Tuned' model to MLflow
        # CORRECTION: Added 'as run' and moved promotion logic INSIDE this block
        with mlflow.start_run(run_name="Tuned_XGBoost_Candidate_Plus_FE") as run:
            mlflow.log_params(self.best_params)
            mlflow.log_param("model_type", "XGBoost_FE")
            mlflow.log_metric("rmse", self.rmse)
            
            final_model = self.build_xgboost_model(**self.best_params)
            final_model.fit(train_df[features], train_df[target])
            mlflow.sklearn.log_model(final_model, artifact_path="model")
                     
            print(f"Logging complete! RMSE {self.rmse:.4f} is now in MLflow.")

            # --- PROMOTION LOGIC (Now safe inside the run) ---
            client = MlflowClient()
            improvement_threshold = 0.01 
            
            if self.rmse < self.champion_rmse * (1 - improvement_threshold):
                print(f"Candidate ({self.rmse:.4f}) beat Champion ({self.champion_rmse:.4f})!")
                
                # CORRECTION: Fixed run_id reference
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
        # Final status report
        status = "RE-TRAINED & UPDATED" if getattr(self, 'retrain_needed', False) else "MONITORED - NO CHANGE"
        print (f"Success! Flow status: {status}")

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
    TaxiMonitoringFlow()

       

    

