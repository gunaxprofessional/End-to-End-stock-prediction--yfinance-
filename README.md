# Stock Price Prediction - End-to-End MLOps Pipeline

An end-to-end ML pipeline for stock price prediction with **feature store**, **model registry**, and **model monitoring**.

## Key Features

- **Feast Feature Store**: Centralized feature management for training and serving
- **MLflow**: Model tracking, versioning, and registry
- **Evidently AI**: Data drift and model performance monitoring
- **SHAP**: Model explainability with feature importance
- **FastAPI**: REST APIs for model serving

## Models

| Model | Objective | Port |
|-------|-----------|------|
| **Price Regressor** | Predict next day closing price | 8001 |
| **Direction Classifier** | Predict UP/DOWN movement | 8002 |

## Project Structure

```
project/
├── config/
│   └── global_config.py         # All configuration paths
│
├── feature_store/
│   ├── feature_repo/
│   │   ├── definitions.py       # Feast feature definitions
│   │   └── feature_store.yaml   # Feast config
│   └── materialize_features.py  # Materialize to online store
│
├── pipelines/
│   ├── data_ingestion.py        # Fetch stock data
│   └── feature_engineering.py   # Create features
│
├── models/
│   ├── price_regression/
│   │   ├── config.py            # Model config
│   │   ├── train.py             # Training pipeline
│   │   ├── serve.py             # FastAPI service
│   │   ├── update_actuals.py    # Update with actual values
│   │   └── monitor.py           # Drift & performance reports
│   │
│   └── direction_classifier/
│       ├── config.py
│       ├── train.py
│       ├── serve.py
│       ├── update_actuals.py
│       └── monitor.py
│
├── artifacts/                    # All generated files (gitignored)
│   ├── data/
│   │   ├── raw/                 # Raw stock data
│   │   ├── processed/           # Processed features
│   │   └── predictions/         # Model predictions
│   ├── feast/                   # Feast registry & online store
│   ├── mlruns/                  # MLflow runs & models
│   └── reports/                 # Evidently reports
│
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
python pipelines/data_ingestion.py
python pipelines/feature_engineering.py
python feature_store/materialize_features.py
```

### 3. Train Models
```bash
python models/price_regression/train.py
python models/direction_classifier/train.py
```

### 4. Serve Models
```bash
# Terminal 1
python models/price_regression/serve.py      # Port 8001

# Terminal 2
python models/direction_classifier/serve.py  # Port 8002
```

### 5. Update Actuals & Monitor
```bash
# After actual values are available
python -m models.price_regression.update_actuals
python -m models.direction_classifier.update_actuals

# Generate monitoring reports
python -m models.price_regression.monitor
python -m models.direction_classifier.monitor
```

## API Endpoints

### Price Regression (Port 8001)
```bash
# Root
curl http://localhost:8001/

# Predict next closing price for all stocks
curl http://localhost:8001/predict_next_close
```

### Direction Classifier (Port 8002)
```bash
# Root
curl http://localhost:8002/

# Predict direction for all stocks
curl http://localhost:8002/predict_direction

# Predict direction for single stock
curl http://localhost:8002/predict_direction/AAPL
```

## Pipeline Flow

```
1. Data Ingestion     → Fetch from yfinance → artifacts/data/raw/
2. Feature Engineering → Create features    → artifacts/data/processed/
3. Materialize        → Push to Feast      → artifacts/feast/
4. Train              → MLflow tracking    → artifacts/mlruns/
5. Serve              → FastAPI + Feast    → Predictions API
6. Update Actuals     → Match predictions  → Same predictions file
7. Monitor            → Evidently reports  → artifacts/reports/
```

## Configuration

All paths and settings are in `config/global_config.py`:

- `STOCKS`: List of stock tickers to track
- `START_DATE`: Data fetch start date
- `PREDICTION_DATE`: Date for predictions
- `ARTIFACTS_DIR`: Base directory for all outputs

## Tech Stack

| Component | Technology |
|-----------|------------|
| Feature Store | Feast |
| ML Tracking | MLflow |
| Monitoring | Evidently AI |
| Explainability | SHAP |
| API | FastAPI |
| Data Source | yfinance |

## Stocks Tracked

- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- AMZN (Amazon)
- TSLA (Tesla)
