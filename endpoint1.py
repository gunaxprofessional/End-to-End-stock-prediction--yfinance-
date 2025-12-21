import fastapi
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient

# CONFIG
MODEL_NAME = "StockPricePredictor"
MODEL_ALIAS = "champion"
TRACKING_URI = "sqlite:///mlflow.db"

DATA_PATH = "data/processed/stock_data_processed.csv"
LATEST_DATE = "2025-11-28"

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

# CREATE SHAP EXPLAINER (ONCE)
explainer = shap.TreeExplainer(
    model,
    shap_background
)

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
        return {"error": "No data found for latest date"}

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
            "prediction": float(predictions[i]),
            "shap_values": {
                feature_cols[j]: float(shap_values[i][j])
                for j in range(len(feature_cols))
            }
        }

    return response


# LOCAL RUN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
