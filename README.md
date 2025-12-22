# Stock Price Prediction

End-to-end ML pipeline for stock price prediction using yfinance data.

## Architecture

```
yfinance API
     │
     ▼
┌─────────────────┐
│ Data Ingestion  │ ──► data/raw/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Eng.    │ ──► data/processed/
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │ ──► MLflow (mlruns/)
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│ FastAPI Server  │ ◄──► │ Update Actuals  │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ▼                        ▼
    Predictions ──────────► data/predictions/
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ Evidently       │ ──► Monitoring_Reports/
                          └─────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

1. **Data Ingestion** - Fetch stock data from yfinance
   ```bash
   python data_ingestion.py
   ```

2. **Feature Engineering** - Generate technical indicators
   ```bash
   python feature_engineering.py
   ```

3. **Model Training** - Train XGBoost model with MLflow tracking
   ```bash
   python model_building.py
   ```

4. **Inference API** - FastAPI endpoint for predictions
   ```bash
   uvicorn predict_endpoint:app --reload
   ```

5. **Update Actuals** - Backfill actual values for past predictions
   ```bash
   python update_actuals.py
   ```

6. **Monitoring** - Generate drift and performance reports
   ```bash
   python monitoring_evidently.py
   ```

## Project Structure

```
data/               Raw and processed stock data
mlruns/             MLflow experiment tracking
Monitoring_Reports/ Evidently HTML reports
```
