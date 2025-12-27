import fastapi
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient
from datetime import timedelta, datetime
from pathlib import Path
import os

# CONFIG
MODEL_NAME = "StockPricePredictor"
MODEL_ALIAS = "champion"
TRACKING_URI = "sqlite:///mlflow.db"

DATA_PATH = Path("data") / "processed" / "stock_data_processed.csv"

# LATEST_DATE = pd.to_datetime(datetime.now().date()) - timedelta(days=1)  # yesterday date
LATEST_DATE = pd.to_datetime("2025-12-24",format="%Y-%m-%d")

# APP INIT
app = fastapi.FastAPI()
mlflow.set_tracking_uri(TRACKING_URI)

# LOAD MODEL
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

# LOAD SHAP BACKGROUND
client = MlflowClient()

model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    MODEL_ALIAS
)

run_id = model_version.run_id

background_path = client.download_artifacts(
    run_id,
    "shap/shap_background.parquet"
)

shap_background = pd.read_parquet(background_path)

# CREATE SHAP EXPLAINER
explainer = shap.TreeExplainer(
    model,
    shap_background
)

@app.get("/")
def read_root():
    return {"welcome": "welcome to stock price prediction api"}

# API ENDPOINT
@app.get("/predict_next_close")
def predict_next_close():
    """
    Predict next close price and return SHAP explanations
    """

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])

    latest_rows = data[data["Date"] == pd.to_datetime(LATEST_DATE)]

    if latest_rows.empty:
        return {"error": f"No data found for latest date: {LATEST_DATE.date()}, Data available till {data['Date'].max().date()}"}

    feature_cols = [
        c for c in latest_rows.columns
        if c not in ["Date", "Ticker", "Target"]
    ]

    X_latest = latest_rows[feature_cols].astype(float)
    tickers = latest_rows["Ticker"].tolist()


    # PREDICTION (SKLEARN)
    predictions = model.predict(X_latest)

    # SHAP VALUES
    shap_values = explainer.shap_values(X_latest)

    response = {}
    for i, ticker in enumerate(tickers):
        response[ticker] = {
            "date": LATEST_DATE.date(),
            "prediction": float(predictions[i]),
            "shap_values": {
                feature_cols[j]: float(shap_values[i][j])
                for j in range(len(feature_cols))
            }
        }

    # logging prediction
    prediction_df = pd.DataFrame({
        "Date": LATEST_DATE.date(),
        "Ticker": tickers,
        "Prediction": predictions,
        "Actual": None
    })
    
    # Add input features to prediction_df
    prediction_df = pd.concat([prediction_df, X_latest.reset_index(drop=True)], axis=1)

    # Save Prediction DataFrame
    pred_path = Path("data") / "predictions" / "predictions.csv"
    if os.path.exists(pred_path):
        existing_df = pd.read_csv(pred_path)
        prediction_df = pd.concat([existing_df, prediction_df], ignore_index=True)
    
    prediction_df.to_csv(pred_path, index=False)

    # 2. SHAP DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "Ticker", tickers)
    shap_df.insert(0, "Date", LATEST_DATE.date())

    # Save SHAP DataFrame
    shap_path = Path("data") / "predictions" / "shap_values.csv"
    if os.path.exists(shap_path):
        existing_shap = pd.read_csv(shap_path)
        shap_df = pd.concat([existing_shap, shap_df], ignore_index=True)
    
    shap_df.to_csv(shap_path, index=False)

    return response


# LOCAL RUN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
