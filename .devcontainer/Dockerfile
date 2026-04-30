FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies for Metaflow and MLflow
RUN apt-get update && apt-get install -y git

# Install your project libraries
RUN pip install metaflow mlflow pandas pyarrow scikit-learn nannyml xgboost optuna torch