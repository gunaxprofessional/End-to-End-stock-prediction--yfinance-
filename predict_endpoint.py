import fastapi
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient
from datetime import timedelta, datetime
from pathlib import Path
import os
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")

# Define features to retrieve from Feast (same as in model_building.py)
FEAST_FEATURES = [
    "stock_features:Close", "stock_features:High",
    "stock_features:Low", "stock_features:Open",
    "stock_features:Volume", "stock_features:Returns",
    "stock_features:High_Low_Pct", "stock_features:Close_Open_Pct",
    "stock_features:MA_3", "stock_features:MA_6", "stock_features:MA_8",
    "stock_features:Volatility_3", "stock_features:Volatility_6",
    "stock_features:Volume_MA_3", "stock_features:Volume_Ratio"
]

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

    # Get unique tickers from the data to query Feast
    tickers = data["Ticker"].unique().tolist()
    
    # Retrieve online features from Feast
    print(f"Retrieving online features for {len(tickers)} tickers from Feast...")
    entity_rows = [{"ticker": ticker} for ticker in tickers]
    
    online_features = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=entity_rows
    ).to_dict()

    # Convert to DataFrame and fix column names
    X_latest_df = pd.DataFrame(online_features)
    
    # Map Feast feature names (stock_features:Feature) back to simple names (Feature)
    # The model expects simple names like 'Close', 'MA_3', etc.
    feature_map = {f: f.split(':')[-1] for f in FEAST_FEATURES}
    X_latest_df = X_latest_df.rename(columns=feature_map)

    # Reorder columns to match the features used during training
    # Note: excluding 'ticker' as it's the entity key
    feature_cols = [f.split(':')[-1] for f in FEAST_FEATURES]
    X_latest = X_latest_df[feature_cols].astype(float)


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
    uvicorn.run(app, host="127.0.0.1", port=8001)
