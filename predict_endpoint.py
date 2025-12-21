import fastapi
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.pyfunc
import logging

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = fastapi.FastAPI()
model = mlflow.pyfunc.load_model("models:/StockPricePredictor@champion")

@app.get("/predict_next_close")
def predict_next_close():
    # Load processed data
    data = pd.read_csv("data/processed/stock_data_processed.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    latest_date = pd.to_datetime('2025-11-28')
    latest_rows = data[data['Date'] == latest_date]

    # Drop columns not used for prediction
    feature_cols = [col for col in latest_rows.columns if col not in ['Date', 'Ticker', 'Target']]
    X_latest = latest_rows[feature_cols].astype('float64')
    tickers = latest_rows['Ticker'].tolist()
    preds = model.predict(X_latest)
    return {ticker: pred for ticker, pred in zip(tickers, preds)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)