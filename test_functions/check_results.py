import mlflow

# Point to your local runs folder
mlflow.set_tracking_uri("file:///workspaces/Capstone_Project/mlruns")

# Get the experiment
experiment = mlflow.get_experiment_by_name("green_taxi_tip_model")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print("\n--- MLFLOW DATA FOUND ---")
for index, run in runs.iterrows():
    print(f"Run Name: {run['tags.mlflow.runName']}")
    print(f"RMSE: {run['metrics.rmse']:.4f}")
    print(f"Model Type: {run['params.model_type']}")
    print("--------------------------\n")