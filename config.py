import os
from pathlib import Path

PROJECT_NAME = "stock_prediction"
TEAM_NAME = "Alpha-Quant"

STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
START_DATE = "2024-12-22"

# MinIO credentials from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "stock-prediction")

if not MINIO_ENDPOINT:
    raise ValueError("MINIO_ENDPOINT environment variable is required")
if not MINIO_ACCESS_KEY:
    raise ValueError("MINIO_ACCESS_KEY environment variable is required")
if not MINIO_SECRET_KEY:
    raise ValueError("MINIO_SECRET_KEY environment variable is required")

# MinIO object keys
RAW_DATA_KEY = "data/raw/stock_data.csv"
PROCESSED_DATA_KEY = "data/processed/stock_data_processed.csv"
FEATURE_PARQUET_KEY = "data/processed/stock_features.parquet"
PREDICTIONS_KEY = "data/predictions/predictions.csv"
PREDICTIONS_ACTUALS_KEY = "data/predictions/predictions_with_actuals.csv"
SHAP_VALUES_KEY = "data/predictions/shap_values.csv"
REGISTRY_KEY = "data/registry.db"
ONLINE_STORE_KEY = "data/online_store.db"

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

MODEL_NAME = "StockPricePredictor"
EXPERIMENT_NAME = "Stock_Price_Prediction"
TRACKING_URI = "sqlite:///mlflow.db"

TRAIN_START_DATE = '2024-11-01'
TRAIN_END_DATE = '2025-08-31'
TEST_START_DATE = '2025-09-01'
TEST_END_DATE = '2025-10-31'

MODEL_ALIAS = "champion"
PREDICTION_DATE = "2025-12-24"

REPORTS_DIR = Path("Monitoring_Reports")
DRIFT_REPORT_KEY = "reports/data_drift_report.html"
REGRESSION_REPORT_KEY = "reports/regression_performance_report.html"
