Taxi Fare Monitoring Capstone
This project implements an end-to-end machine learning pipeline for predicting and monitoring New York City "Green" Taxi tips. It transitions from a baseline model to an optimized production-ready champion, with integrated data drift monitoring and performance estimation using Metaflow for orchestration, MLflow for experiment tracking and registry, and Evidently AI for production monitoring.

## MLflow Model Registry & Project Overview
The pipeline follows a four-stage evolution, with the model registered across four versions to demonstrate progress and adaptation to new data:

### 1. Baseline Model (01_baseline_decision_tree.py)
Model: Decision Tree Regressor.

Role: Established the initial performance baseline, registered as Version 1 using January data.

### 2. Optimized XGBoost (02_xgboost_simple.py)
Model: XGBoost with Optuna hyperparameter tuning.

Role: Replaced underperforming default settings with an automated optimization framework to ensure a competitive Version 2.

### 3. Feature-Enriched Champion (03_xgboost_optimized_champion.py)
Model: Optimized XGBoost + Advanced Feature Engineering.

Role: Builds upon the tuned model by introducing additional calculated features derived from raw data.

Registry Progression:

Version 3: Optimized using January reference data and an April batch.

Version 4: Optimized using January reference data and an August batch.

Logic: Implements a "Champion Gate" to ensure only models that improve the current RMSE are registered as the new production @champion.

### 4. Data & Performance Monitoring (04_taxi_flow_monitoring.py)
Role: Monitors the production environment using August data as a test batch.

Analysis: Uses Evidently AI to compare the production batch against the training reference to detect data drift.

Outputs: Generates interactive HTML reports: univariate_drift.html and performance_estimation.html.

## Data Setup
Note for Reviewers: The TLC_data/ folder is ignored by Git due to file size. To reproduce these results:

Create a TLC_data/ directory in the project root.

Download the NYC Green Taxi Parquet files (January, April, and August 2020) into that folder.

## Installation & Execution
All commands should be executed from the development/ directory.

### 1. Start the MLflow Server

Bash
mlflow ui --backend-store-uri file:///workspaces/Taxi_Fare_Monitoring_Capstone/mlruns --port 5000

### 2. Run the Pipeline
To avoid overwriting existing registry versions, reviewers can use a unique model name:

Bash
# Example: Training the August Champion
python 03_xgboost_optimized_champion.py run \
  --model-name reviewer_test_model \
  --reference-path ../TLC_data/green_tripdata_2020-01.parquet \
  --batch-path ../TLC_data/green_tripdata_2020-08.parquet

# Running the Monitoring Flow
python 04_taxi_flow_monitoring.py run \
  --reference-path ../TLC_data/green_tripdata_2020-01.parquet \
  --batch-path ../TLC_data/green_tripdata_2020-08.parquet