# Databricks notebook source
# MAGIC %pip install xgboost mlflow joblib  # If not already
# MAGIC
# MAGIC %run ./src/train.py  # This loads data, trains XGBoost, logs to MLflow, dumps PKLs to models/
