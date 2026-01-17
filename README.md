# Stock Price Prediction - End-to-End MLOps Pipeline

An end-to-end ML pipeline for stock price prediction with **Docker containerization**, **Airflow orchestration**, **feature store**, **model registry**, and **model monitoring**.

## Key Features

- **Docker**: Containerized ML pipeline with one container per model
- **Apache Airflow 3.0**: DAG-based orchestration for automated pipelines
- **Feast Feature Store**: Centralized feature management for training and serving
- **MLflow**: Model tracking, versioning, and registry
- **Evidently AI**: Data drift and model performance monitoring
- **SHAP**: Model explainability with feature importance
- **FastAPI**: REST APIs for model serving

## Models

| Model                    | Objective                      | Port |
| ------------------------ | ------------------------------ | ---- |
| **Price Regressor**      | Predict next day closing price | 8001 |
| **Direction Classifier** | Predict UP/DOWN movement       | 8002 |

## Project Structure

```
project/
├── config/
│   └── global_config.py          # Configuration with env var support
│
├── feature_store/
│   ├── feature_repo/
│   │   ├── definitions.py        # Feast feature definitions
│   │   └── feature_store.yaml    # Feast config
│   └── materialize_features.py   # Materialize to online store
│
├── pipelines/
│   ├── data_ingestion.py         # Fetch stock data
│   └── feature_engineering.py    # Create features
│
├── models/
│   ├── price_regression/
│   │   ├── config.py             # Model config
│   │   ├── train.py              # Training pipeline
│   │   ├── serve.py              # FastAPI service
│   │   ├── update_actuals.py     # Update with actual values
│   │   └── monitor.py            # Drift & performance reports
│   │
│   └── direction_classifier/
│       ├── config.py
│       ├── train.py
│       ├── serve.py
│       ├── update_actuals.py
│       └── monitor.py
│
├── docker/
│   ├── base/Dockerfile
│   ├── data-ingestion/Dockerfile
│   ├── feature-engineering/Dockerfile
│   ├── price-regression/
│   │   ├── Dockerfile.train
│   │   └── Dockerfile.serve
│   └── direction-classifier/
│       ├── Dockerfile.train
│       └── Dockerfile.serve
│
├── airflow-server/
│   ├── dags/
│   │   ├── data_pipeline_dag.py
│   │   ├── price_regression_dag.py
│   │   ├── direction_classifier_dag.py
│   │   ├── price_regression_monitor_dag.py
│   │   └── direction_classifier_monitor_dag.py
│   ├── docker-compose.yaml       # Airflow stack
│   └── .env
│
├── requirements/
│   ├── base.txt
│   ├── training.txt
│   └── serving.txt
│
├── docker-compose.yaml           # ML containers
├── .dockerignore
│
└── artifacts/                    # All generated files (gitignored)
    ├── data/
    │   ├── raw/                  # Raw stock data
    │   ├── processed/            # Processed features
    │   └── predictions/          # Model predictions
    ├── feast/                    # Feast registry & online store
    ├── mlruns/                   # MLflow runs & models
    └── reports/                  # Evidently reports
```

## Quick Start

### Option 1: Docker (Recommended)

#### Build Images

```bash
docker-compose build
```

#### Run Pipeline Manually

```bash
# Create network and volume
docker network create stock-network
docker volume create stock-artifacts

# Run pipeline
docker-compose run --rm data-ingestion
docker-compose run --rm feature-engineering
docker-compose run --rm price-trainer
docker-compose run --rm direction-trainer

# Start servers
docker-compose up -d price-server direction-server
```

#### Run with Airflow (Automated)

```bash
# Start Airflow
cd airflow-server
docker-compose up -d

# Access Airflow UI: http://localhost:8080

# Trigger DAGs manually
docker exec airflow-server-airflow-scheduler-1 airflow dags trigger data_pipeline
```

### Option 2: Local Development

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run Data Pipeline

```bash
python pipelines/data_ingestion.py
python pipelines/feature_engineering.py
python feature_store/materialize_features.py
```

#### Train Models

```bash
python models/price_regression/train.py
python models/direction_classifier/train.py
```

#### Serve Models

```bash
# Terminal 1
python models/price_regression/serve.py      # Port 8001

# Terminal 2
python models/direction_classifier/serve.py  # Port 8002
```

## Airflow DAGs

| DAG                            | Schedule      | Description                             |
| ------------------------------ | ------------- | --------------------------------------- |
| `data_pipeline`                | 5 AM weekdays | Data ingestion → Feature engineering    |
| `price_regression_model`       | 6 AM weekdays | Train price regression model            |
| `direction_classifier_model`   | 6 AM weekdays | Train direction classifier model        |
| `price_regression_monitor`     | 6 PM weekdays | Update actuals → Generate drift reports |
| `direction_classifier_monitor` | 6 PM weekdays | Update actuals → Generate drift reports |

### Daily Pipeline Flow

```
05:00 AM  → data_pipeline
              ├── data_ingestion
              └── feature_engineering

06:00 AM  → price_regression_model (waits for data_pipeline)
          → direction_classifier_model (waits for data_pipeline)

          → Servers make predictions throughout the day

06:00 PM  → price_regression_monitor
              ├── update_actuals
              └── run_monitor (drift reports)
          → direction_classifier_monitor
              ├── update_actuals
              └── run_monitor (drift reports)
```

## Docker Containers

| Container           | Image                     | Purpose                                     |
| ------------------- | ------------------------- | ------------------------------------------- |
| data-ingestion      | stock-data-ingestion      | Fetch stock data                            |
| feature-engineering | stock-feature-engineering | Feature engineering + Feast materialization |
| price-trainer       | stock-price-trainer       | Train price regression model                |
| direction-trainer   | stock-direction-trainer   | Train direction classifier                  |
| price-server        | stock-price-server        | Serve price predictions (port 8001)         |
| direction-server    | stock-direction-server    | Serve direction predictions (port 8002)     |

## API Endpoints

### Price Regression (Port 8001)

```bash
curl http://localhost:8001/
curl http://localhost:8001/health
curl http://localhost:8001/predict_next_close
```

### Direction Classifier (Port 8002)

```bash
curl http://localhost:8002/
curl http://localhost:8002/health
curl http://localhost:8002/predict_direction
curl http://localhost:8002/predict_direction/AAPL
```

## Architecture

```mermaid
graph LR
    %% Data Sources
    YF[Yahoo Finance]
    
    %% Orchestration
    AIRFLOW[Airflow<br/>Scheduler]
    
    %% Pipeline Stages
    INGEST[Data<br/>Ingestion]
    FEATURES[Feature<br/>Engineering]
    FEAST[(Feast<br/>Feature Store)]
    
    %% Training
    TRAIN[Model<br/>Training]
    MLFLOW[(MLflow<br/>Registry)]
    
    %% Serving
    API[Prediction<br/>APIs<br/>:8001/:8002]
    
    %% Monitoring
    MONITOR[Monitoring<br/>& Reports]
    
    %% Main Flow
    YF -->|Stock Data| INGEST
    AIRFLOW -.->|Orchestrates| INGEST
    INGEST --> FEATURES
    FEATURES --> FEAST
    
    FEAST -->|Offline Features| TRAIN
    TRAIN -->|Log & Register| MLFLOW
    
    MLFLOW -->|Load Models| API
    FEAST -->|Online Features| API
    
    API -->|Predictions| MONITOR
    YF -->|Actual Prices| MONITOR
    AIRFLOW -.->|Schedule| MONITOR
    
    %% Styling
    classDef source fill:#e1f5ff,stroke:#333,stroke-width:2px,color:black
    classDef orchestration fill:#fff3cd,stroke:#333,stroke-width:2px,color:black
    classDef pipeline fill:#d4edda,stroke:#333,stroke-width:2px,color:black
    classDef storage fill:#d1ecf1,stroke:#333,stroke-width:2px,color:black
    classDef model fill:#cce5ff,stroke:#333,stroke-width:2px,color:black
    classDef serve fill:#f8d7da,stroke:#333,stroke-width:2px,color:black
    classDef monitor fill:#e2e3e5,stroke:#333,stroke-width:2px,color:black
    
    class YF source
    class AIRFLOW orchestration
    class INGEST,FEATURES pipeline
    class FEAST,MLFLOW storage
    class TRAIN model
    class API serve
    class MONITOR monitor
```

### Pipeline Flow

**1. Data Pipeline** (Orchestrated by Airflow)
- Fetch stock data from Yahoo Finance
- Engineer 15 technical indicators
- Materialize features to Feast (offline + online stores)

**2. Training Pipeline**
- Retrieve historical features from Feast offline store
- Train XGBoost models (Price Regression + Direction Classifier)
- Log experiments and register models in MLflow as `@champion`

**3. Serving Pipeline**
- Load `@champion` models from MLflow registry
- Fetch real-time features from Feast online store
- Serve predictions via FastAPI (ports 8001, 8002)
- Generate SHAP explanations

**4. Monitoring Pipeline**
- Collect predictions and actual outcomes
- Generate Evidently AI drift and performance reports
- Store HTML reports in artifacts volume

**Infrastructure:** All components run in Docker containers sharing the `stock-artifacts` volume

## Tech Stack

| Component        | Technology         |
| ---------------- | ------------------ |
| Containerization | Docker             |
| Orchestration    | Apache Airflow 3.0 |
| Feature Store    | Feast              |
| ML Tracking      | MLflow             |
| Monitoring       | Evidently AI       |
| Explainability   | SHAP               |
| API              | FastAPI            |
| Data Source      | yfinance           |

## Stocks Tracked

- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- AMZN (Amazon)
- TSLA (Tesla)

## Configuration

All paths and settings are in `config/global_config.py`:

- `STOCKS`: List of stock tickers to track
- `START_DATE`: Data fetch start date
- `PREDICTION_DATE`: Date for predictions
- `APP_DIR`: Base application directory (env var supported)
- `ARTIFACTS_DIR`: Base directory for all outputs (env var supported)
