# Docker Integration

This document outlines the Docker containerization setup for the stock prediction pipeline.

## Overview

The project is containerized using Docker to ensure consistent execution across different environments. Each pipeline stage runs in its own isolated container, sharing data through persistent volumes.

## Container Structure

```
docker/
├── base/                      # Base image with common dependencies
├── data-ingestion/            # Stock data collection
├── feature-engineering/       # Feature creation + Feast materialization
├── direction-classifier/      # Binary classification (train + serve)
└── price-regression/          # Price prediction (train + serve)
```

## Containers Implemented

### 1. Base Image
**Path:** `docker/base/Dockerfile`

Foundation image containing:
- Python 3.12-slim
- System dependencies (gcc, g++)
- Base requirements (pandas, numpy, feast, mlflow, yfinance)
- Common code (config, pipelines, feature_store, models)

### 2. Data Ingestion
**Path:** `docker/data-ingestion/Dockerfile`

- Downloads stock data from Yahoo Finance
- Command: `python pipelines/data_ingestion.py`
- Output: `artifacts/data/raw/stock_data.csv`

### 3. Feature Engineering
**Path:** `docker/feature-engineering/Dockerfile`

- Creates technical indicators
- Materializes features to Feast online store
- Command: `python pipelines/feature_engineering.py && python feature_store/materialize_features.py`
- Outputs: 
  - `artifacts/data/processed/stock_features.parquet`
  - Feast online store populated

### 4. Direction Classifier

**Training:** `docker/direction-classifier/Dockerfile.train`
- Trains XGBoost binary classifier
- Command: `python models/direction_classifier/train.py`
- Registers model in MLflow

**Serving:** `docker/direction-classifier/Dockerfile.serve`
- FastAPI endpoint on port 8002
- Command: `python models/direction_classifier/serve.py`
- Endpoints: `/health`, `/predict`

### 5. Price Regression

**Training:** `docker/price-regression/Dockerfile.train`
- Trains XGBoost regression model
- Command: `python models/price_regression/train.py`
- Registers model in MLflow

**Serving:** `docker/price-regression/Dockerfile.serve`
- FastAPI endpoint on port 8001
- Command: `python models/price_regression/serve.py`
- Endpoints: `/health`, `/predict`

## Docker Compose Setup

The `docker-compose.yaml` orchestrates all containers:

**Shared Resources:**
- **Volume:** `stock-artifacts` - Persistent storage for data, models, and reports
- **Network:** `stock-network` - Enables container communication

**Service Dependencies:**
```
data-ingestion
    ↓
feature-engineering
    ↓
[price-trainer, direction-trainer]
    ↓
[price-server:8001, direction-server:8002]
```

## Environment Variables

All containers use:
```bash
APP_DIR=/app
ARTIFACTS_DIR=/app/artifacts
PYTHONUNBUFFERED=1
```

These ensure paths work correctly in containerized environments.

## Steps to Run

### 1. Create Network (First Time Only)
```bash
docker network create stock-network
```

### 2. Run Data Pipeline
```bash
# Fetch stock data
docker-compose run data-ingestion

# Create features and materialize to Feast
docker-compose run feature-engineering
```

### 3. Train Models
```bash
# Train price regression model
docker-compose run price-trainer

# Train direction classifier
docker-compose run direction-trainer
```

### 4. Start Prediction APIs
```bash
# Start both serving containers
docker-compose up price-server direction-server

# Or run in detached mode
docker-compose up -d price-server direction-server
```

### 5. Test APIs
```bash
# Test price prediction API
curl http://localhost:8001/health

# Test direction prediction API
curl http://localhost:8002/health
```

## Volume Structure

The `stock-artifacts` volume contains:

```
/app/artifacts/
├── data/
│   ├── raw/                 # Original stock data
│   ├── processed/           # Engineered features (parquet)
│   └── predictions/         # Model predictions
├── feast/
│   ├── registry.db          # Feast metadata
│   └── online_store.db      # Online features
├── mlruns/                  # MLflow experiments and models
└── reports/                 # Monitoring reports
```

## Dockerfile Pattern

All containers follow this structure:

1. **Base OS:** `python:3.12-slim`
2. **System deps:** gcc, g++ (for compilation)
3. **Python deps:** Install from requirements files
4. **ENV vars:** Set APP_DIR and ARTIFACTS_DIR
5. **Code:** Copy relevant modules
6. **CMD:** Run specific script

Training and serving use separate Dockerfiles to optimize dependencies.

## Key Implementation Details

### Container-Aware Paths

Code uses environment variables for flexibility:

```python
BASE_DIR = Path(os.getenv("APP_DIR", Path(__file__).parent.parent))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR / "artifacts"))
```

This allows the same code to run locally and in containers.

### Requirements Separation

- `requirements/base.txt` - Core dependencies (all containers)
- `requirements/training.txt` - ML training libraries
- `requirements/serving.txt` - FastAPI, uvicorn

### Health Checks

Serving containers include health checks:
- Interval: 30s
- Timeout: 10s
- Retries: 3

## Common Commands

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build price-server

# View logs
docker-compose logs -f price-server

# Stop all services
docker-compose down

# Clean up volumes (⚠️ deletes data)
docker-compose down -v

# Run container with shell (debugging)
docker-compose run --rm data-ingestion bash
```

## Integration Benefits

✅ **Reproducibility** - Same environment everywhere  
✅ **Isolation** - Each component is independent  
✅ **Scalability** - Easy to deploy to cloud platforms  
✅ **Version Control** - Infrastructure as code  
✅ **CI/CD Ready** - Automated builds and deployments  

## Notes

- All containers share the `stock-artifacts` volume for data persistence
- Network must be created before first run
- Training containers run once and exit
- Serving containers run continuously with auto-restart
- Volumes persist data between container restarts
