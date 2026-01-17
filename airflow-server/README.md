# Airflow Server

Apache Airflow 3.0.1 setup for orchestrating the ML pipeline using DockerOperator.

## Overview

This setup uses the **official Airflow docker-compose.yaml** with minimal modifications to support DockerOperator for running ML containers.

## Changes from Official docker-compose.yaml

The following changes were made to the [official Airflow 3.0.1 docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/3.0.1/docker-compose.yaml):

| Line | Change | Purpose |
|------|--------|---------|
| 63 | `AIRFLOW__CORE__LOAD_EXAMPLES: 'false'` | Disable example DAGs |
| 72 | `_PIP_ADDITIONAL_REQUIREMENTS: apache-airflow-providers-docker` | Install Docker provider for DockerOperator |
| 73 | `DOCKER_HOST: unix:///var/run/docker.sock` | Docker socket environment variable |
| 81 | `/var/run/docker.sock:/var/run/docker.sock` | Mount Docker socket for DockerOperator |
| 337-340 | External `stock-network` network | Connect to ML containers network |

## Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| Airflow Web UI | `airflow` | `airflow` |
| PostgreSQL | `airflow` | `airflow` |

> **Note:** These are default development credentials. For production, set custom credentials via environment variables in `.env` file.

## Environment Variables

Create a `.env` file in this directory to customize settings:

```bash
# Airflow image (optional)
AIRFLOW_IMAGE_NAME=apache/airflow:3.0.1

# User ID for file permissions (Linux only)
AIRFLOW_UID=50000

# Custom admin credentials
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=your_secure_password

# Additional pip packages (optional, comma-separated)
_PIP_ADDITIONAL_REQUIREMENTS=apache-airflow-providers-docker
```

## Quick Start

### Prerequisites

1. Docker and Docker Compose installed
2. Create the external network and volume (if not exists):
   ```bash
   docker network create stock-network
   docker volume create stock-artifacts
   ```
3. Build ML container images (from project root):
   ```bash
   docker-compose build
   ```

### Start Airflow

```bash
# Create required directories
mkdir -p dags logs plugins config

# Start all services (includes automatic DB initialization)
docker-compose up -d

# Check service health
docker-compose ps
```

> **Note:** Database initialization (`airflow db migrate`) and admin user creation happen automatically via the `airflow-init` service. The environment variables `_AIRFLOW_DB_MIGRATE=true` and `_AIRFLOW_WWW_USER_CREATE=true` handle this. No manual `airflow db init` is needed.

### Access Web UI

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

## Services

| Service | Description | Port |
|---------|-------------|------|
| `postgres` | Metadata database | 5432 (internal) |
| `redis` | Celery message broker | 6379 (internal) |
| `airflow-apiserver` | Web UI & REST API | 8080 |
| `airflow-scheduler` | Task scheduling | - |
| `airflow-dag-processor` | DAG file parsing | - |
| `airflow-worker` | Celery task executor | - |
| `airflow-triggerer` | Deferrable task handler | - |

## DAGs

| DAG | Schedule | Description |
|-----|----------|-------------|
| `data_pipeline` | 5 AM weekdays | Data ingestion → Feature engineering |
| `price_regression_model` | 6 AM weekdays | Train price regression model (waits for data_pipeline) |
| `direction_classifier_model` | 6 AM weekdays | Train direction classifier (waits for data_pipeline) |
| `price_regression_monitor` | 6 PM weekdays | Update actuals → Generate drift reports |
| `direction_classifier_monitor` | 6 PM weekdays | Update actuals → Generate drift reports |

### DAG Dependencies

The model training DAGs use `ExternalTaskSensor` to wait for the data pipeline:

```
data_pipeline (5 AM)
    ├── data_ingestion
    └── feature_engineering
            │
            ▼ (waits with execution_delta=1 hour)
price_regression_model (6 AM)
direction_classifier_model (6 AM)
```

### Daily Pipeline Flow

```
05:00 AM  → data_pipeline
              ├── data_ingestion (fetches stock data)
              └── feature_engineering (creates features, materializes to Feast)

06:00 AM  → price_regression_model (waits for data_pipeline)
              └── train_price_model (trains and registers model in MLflow)
          → direction_classifier_model (waits for data_pipeline)
              └── train_direction_model (trains and registers model in MLflow)

          → Servers make predictions throughout the day (ports 8001, 8002)

06:00 PM  → price_regression_monitor
              ├── update_actuals (updates predictions with actual values)
              └── run_monitor (generates Evidently drift reports)
          → direction_classifier_monitor
              ├── update_actuals
              └── run_monitor
```

## Docker Images Used

| DAG Task | Docker Image |
|----------|--------------|
| `data_ingestion` | `stock-data-ingestion:latest` |
| `feature_engineering` | `stock-feature-engineering:latest` |
| `train_price_model` | `stock-price-trainer:latest` |
| `train_direction_model` | `stock-direction-trainer:latest` |
| Monitor tasks | `stock-price-trainer:latest` / `stock-direction-trainer:latest` |

## Common Commands

```bash
# List DAGs
docker exec airflow-server-airflow-scheduler-1 airflow dags list

# Trigger a DAG manually
docker exec airflow-server-airflow-scheduler-1 airflow dags trigger data_pipeline

# Unpause a DAG (required before scheduled runs)
docker exec airflow-server-airflow-scheduler-1 airflow dags unpause data_pipeline

# Check DAG runs
docker exec airflow-server-airflow-scheduler-1 airflow dags list-runs data_pipeline

# View worker logs (where tasks execute)
docker-compose logs -f airflow-worker

# Stop all services
docker-compose down

# Stop and remove volumes (full reset)
docker-compose down -v
```

## Testing DAGs Manually

To test the full pipeline:

```bash
# 1. Unpause all DAGs
docker exec airflow-server-airflow-scheduler-1 airflow dags unpause data_pipeline
docker exec airflow-server-airflow-scheduler-1 airflow dags unpause price_regression_model
docker exec airflow-server-airflow-scheduler-1 airflow dags unpause direction_classifier_model

# 2. Trigger data pipeline first
docker exec airflow-server-airflow-scheduler-1 airflow dags trigger data_pipeline

# 3. Wait for it to complete, then trigger model training
docker exec airflow-server-airflow-scheduler-1 airflow dags trigger price_regression_model
docker exec airflow-server-airflow-scheduler-1 airflow dags trigger direction_classifier_model
```

Or test containers directly (bypassing Airflow):

```bash
# From project root
docker-compose run --rm data-ingestion
docker-compose run --rm feature-engineering
docker-compose run --rm price-trainer
docker-compose run --rm direction-trainer
```

## Troubleshooting

### Services not starting

```bash
# Check init logs
docker-compose logs airflow-init

# Verify network exists
docker network ls | grep stock-network

# Verify volume exists
docker volume ls | grep stock-artifacts
```

### DAGs not appearing

```bash
# Check DAG processor logs
docker-compose logs airflow-dag-processor

# Verify DAG files are mounted
docker exec airflow-server-airflow-scheduler-1 ls /opt/airflow/dags/

# Check for DAG import errors
docker exec airflow-server-airflow-scheduler-1 airflow dags list-import-errors
```

### DockerOperator failing

```bash
# Verify Docker socket is accessible from worker
docker exec airflow-server-airflow-worker-1 docker ps

# Check worker logs for task errors
docker-compose logs airflow-worker

# Verify ML images are built
docker images | grep stock-
```

### ExternalTaskSensor timing out

The model training DAGs wait for `data_pipeline` with `execution_delta=timedelta(hours=1)`. For manual triggers, you may need to:

1. Trigger `data_pipeline` first and wait for completion
2. Then trigger model training DAGs

Or run containers directly via `docker-compose run` to bypass the sensor.

### Task shows "up_for_reschedule"

This is normal for sensor tasks - they check periodically until the upstream task completes.
