import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import fastapi
import pandas as pd
import requests
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient

from feast import FeatureStore
from config.global_config import (
    TRACKING_URI, FEAST_FEATURES, FEAST_REPO_PATH, FEATURES_LIST,
    PROCESSED_DATA_PATH, PREDICTIONS_PATH, SHAP_VALUES_PATH, PREDICTION_DATE
)
from models.price_regression.config import MODEL_NAME, MODEL_ALIAS

store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
LATEST_DATE = pd.to_datetime(PREDICTION_DATE, format="%Y-%m-%d")

app = fastapi.FastAPI()
mlflow.set_tracking_uri(str(TRACKING_URI))

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

client = MlflowClient()
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

# Create SHAP background from processed data
print("Loading SHAP background data...")
background_df = pd.read_csv(PROCESSED_DATA_PATH)
shap_background = background_df[FEATURES_LIST].dropna().sample(n=min(100, len(background_df)), random_state=42)
explainer = shap.TreeExplainer(model, shap_background)
print("SHAP explainer ready")


def get_llm_summary(shap_dict: dict, prediction: float, ticker: str) -> str:
    """Generate human-readable summary from SHAP values."""
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(feat, val) for feat, val in sorted_features if abs(val) > 0.0001][:5]

    if not top_features:
        return f"The model predicts the next closing price for {ticker} to be {prediction:.2f}."

    feature_explanations = []
    for feature, value in top_features:
        direction = "up" if value > 0 else "down"
        clean_name = feature.replace('_', ' ')

        if 'MA' in feature:
            clean_name = f"{feature.split('_')[-1]}-day moving average"
        elif 'Volatility' in feature:
            clean_name = f"{feature.split('_')[-1]}-day price volatility"
        elif 'Volume' in feature and 'MA' in feature:
            clean_name = "3-day average trading volume"
        elif 'High_Low_Pct' in feature:
            clean_name = "daily price range (High-Low)"

        feature_explanations.append(f"{clean_name} pushed the prediction {direction} (impact: {value:+.4f})")

    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            reasons = ". ".join(feature_explanations[:2])
            return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."

        response = requests.post(
            "https://router.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"},
            json={"inputs": f"Explain: {ticker} at {prediction:.2f}. {'; '.join(feature_explanations)}",
                  "parameters": {"max_length": 150, "min_length": 30, "do_sample": False}},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get('summary_text') or result[0].get('generated_text', '')

        reasons = ". ".join(feature_explanations[:2])
        return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."

    except Exception:
        reasons = ". ".join(feature_explanations[:2])
        return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."


@app.get("/")
def read_root():
    return {"welcome": "Stock Price Prediction API", "model": MODEL_NAME}


@app.get("/predict_next_close")
def predict_next_close():
    """Predict next day closing price for all tickers."""
    data = pd.read_csv(PROCESSED_DATA_PATH)
    tickers = data["Ticker"].unique().tolist()

    entity_rows = [{"ticker": ticker} for ticker in tickers]
    online_features = store.get_online_features(
        features=FEAST_FEATURES, entity_rows=entity_rows
    ).to_dict()

    feature_cols = [f.split(':')[-1] for f in FEAST_FEATURES]
    X_latest = pd.DataFrame(online_features).rename(
        columns={f: f.split(':')[-1] for f in FEAST_FEATURES}
    )[feature_cols].astype(float)

    predictions = model.predict(X_latest)
    shap_values = explainer.shap_values(X_latest)

    response = {}
    for i, ticker in enumerate(tickers):
        ticker_shap = {feature_cols[j]: float(shap_values[i][j]) for j in range(len(feature_cols))}
        response[ticker] = {
            "date": LATEST_DATE.date().isoformat(),
            "prediction": float(predictions[i]),
            "summary": get_llm_summary(ticker_shap, float(predictions[i]), ticker),
            "shap_values": ticker_shap
        }

    # Save predictions locally
    prediction_df = pd.DataFrame({
        "Date": LATEST_DATE.date(), "Ticker": tickers,
        "Prediction": predictions, "Actual": None
    })
    prediction_df = pd.concat([prediction_df, X_latest.reset_index(drop=True)], axis=1)

    try:
        existing = pd.read_csv(PREDICTIONS_PATH)
        prediction_df = pd.concat([existing, prediction_df], ignore_index=True)
    except FileNotFoundError:
        pass
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    # Save SHAP values locally
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "Ticker", tickers)
    shap_df.insert(0, "Date", LATEST_DATE.date())

    try:
        existing_shap = pd.read_csv(SHAP_VALUES_PATH)
        shap_df = pd.concat([existing_shap, shap_df], ignore_index=True)
    except FileNotFoundError:
        pass
    shap_df.to_csv(SHAP_VALUES_PATH, index=False)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
