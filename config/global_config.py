import os
from pathlib import Path

PROJECT_NAME = "stock_prediction"
TEAM_NAME = "Alpha-Quant"

STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
START_DATE = "2024-12-22"

# Base directory - supports both local dev and container environments
# In containers: set APP_DIR=/app, ARTIFACTS_DIR=/app/artifacts
# Local dev: uses project directory as default
BASE_DIR = Path(os.getenv("APP_DIR", Path(__file__).parent.parent))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))

# Data paths
RAW_DATA_PATH = ARTIFACTS_DIR / "data" / "raw" / "stock_data.csv"
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "data" / "processed" / "stock_data_processed.csv"
FEATURE_PARQUET_PATH = ARTIFACTS_DIR / "data" / "processed" / "stock_features.parquet"
PREDICTIONS_PATH = ARTIFACTS_DIR / "data" / "predictions" / "predictions.csv"
PREDICTIONS_ACTUALS_PATH = ARTIFACTS_DIR / "data" / "predictions" / "predictions_with_actuals.csv"
SHAP_VALUES_PATH = ARTIFACTS_DIR / "data" / "predictions" / "shap_values.csv"

# Report paths
DRIFT_REPORT_PATH = ARTIFACTS_DIR / "reports" / "data_drift_report.html"
REGRESSION_REPORT_PATH = ARTIFACTS_DIR / "reports" / "regression_performance_report.html"

# Feast paths
FEAST_REPO_PATH = BASE_DIR / "feature_store" / "feature_repo"
FEAST_DATA_DIR = ARTIFACTS_DIR / "feast"

# MLflow - use file URI for both tracking and artifacts
MLFLOW_DIR = ARTIFACTS_DIR / "mlruns"
TRACKING_URI = MLFLOW_DIR.as_uri()

FEAST_FEATURES = [
    "stock_features:Close", "stock_features:High",
    "stock_features:Low", "stock_features:Open",
    "stock_features:Volume", "stock_features:Returns",
    "stock_features:High_Low_Pct", "stock_features:Close_Open_Pct",
    "stock_features:MA_3", "stock_features:MA_6", "stock_features:MA_8",
    "stock_features:Volatility_3", "stock_features:Volatility_6",
    "stock_features:Volume_MA_3", "stock_features:Volume_Ratio"
]

FEATURES_LIST = [f.split(':')[-1] for f in FEAST_FEATURES]

PREDICTION_DATE = "2025-12-24"

# Create directories
for path in [RAW_DATA_PATH.parent, PROCESSED_DATA_PATH.parent, PREDICTIONS_PATH.parent,
             DRIFT_REPORT_PATH.parent, FEAST_DATA_DIR, MLFLOW_DIR]:
    path.mkdir(parents=True, exist_ok=True)
