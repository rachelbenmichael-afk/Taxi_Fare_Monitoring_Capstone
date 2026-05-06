
###  The "Forking" Instructions (Optional but Recommended)

```markdown
# 🚀 Instructions for Reviewers
To review this project without modifying the original code:
1. **Fork** this repository to your own GitHub account.
2. In your forked repo, click the green **Code** button, select the **Codespaces** tab, and click **Create codespace on main**.
3. Follow the **Data Setup** and **Execution** steps within the Codespace.
```
## Taxi Fare Monitoring Capstone
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
Note for Reviewers: The `TLC_data/` folder is ignored by Git due to file size. To reproduce these results:

1. **Create** a `TLC_data/` directory in the project root.
2. **Download** the NYC Green Taxi Parquet files directly into that folder:
   * [January 2020 Data](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-01.parquet)
   * [April 2020 Data](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-04.parquet)
   * [August 2020 Data](https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-08.parquet)




1. **Fork** this repository to your own GitHub account.
2. In your forked repo, click the green **Code** button, select the **Codespaces** tab, and click **Create codespace on main**.
3. Follow the **Data Setup** and **Execution** steps below within the Codespace.

---
## Environment & Execution
> **Important**: This project is designed to run exclusively on **GitHub Codespaces**. Please do not attempt to run this locally in VS Code.

* **Environment**: The `environment.yml` file is provided in the root. The Codespace automatically configures this environment upon launch. 
* **Conda**: You do **not** need to manually activate conda in the terminal; the environment is pre-configured for the Python interpreter in this workspace.

### 1. Start the MLflow Server
Run this from the project root:
```bash
mlflow ui --backend-store-uri file://$PWD/mlruns --port 5000

```

### 2. Run the Pipeline
All commands must be executed from the root directory. Note the development/ prefix for the scripts and the root-relative paths for the data:
### Example: Training the August Champion

python development/03_xgboost_optimized_champion.py run \
  --model-name reviewer_test_model \
  --reference-path TLC_data/green_tripdata_2020-01.parquet \
  --batch-path TLC_data/green_tripdata_2020-08.parquet

### Running the Monitoring Flow

python development/04_taxi_flow_monitoring.py run \
  --reference-path TLC_data/green_tripdata_2020-01.parquet \
  --batch-path TLC_data/green_tripdata_2020-08.parquet

  ---

