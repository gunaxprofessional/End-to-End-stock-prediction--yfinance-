import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import fastapi
import pandas as pd
import requests
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient
from pathlib import Path

from feast import FeatureStore
from config.global_config import (
    TRACKING_URI, FEAST_FEATURES, FEAST_REPO_PATH, FEATURES_LIST,
    PROCESSED_DATA_PATH, PREDICTIONS_PATH, PREDICTION_DATE
)
from models.direction_classifier.config import MODEL_NAME, MODEL_ALIAS

# Local paths for direction classifier predictions
DIRECTION_PREDICTIONS_PATH = PREDICTIONS_PATH.parent / "direction_predictions.csv"
DIRECTION_SHAP_PATH = PREDICTIONS_PATH.parent / "direction_shap_values.csv"

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
try:
    background_df = pd.read_csv(PROCESSED_DATA_PATH)
    shap_background = background_df[FEATURES_LIST].dropna().sample(n=min(100, len(background_df)), random_state=42)
    explainer = shap.TreeExplainer(model, shap_background)
    print("SHAP explainer ready")
except Exception as e:
    print(f"SHAP background not available: {e}")
    explainer = None


def normalize_shap_values(shap_values, for_class=1):
    """Normalize SHAP values to 2D for binary classification."""
    if isinstance(shap_values, list):
        return shap_values[for_class]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        return shap_values[:, :, for_class]
    return shap_values


def get_llm_summary(shap_dict: dict, direction: str, confidence: float, ticker: str) -> str:
    """Generate human-readable summary from SHAP values."""
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(feat, val) for feat, val in sorted_features if abs(val) > 0.0001][:5]

    if not top_features:
        return f"The model predicts {ticker} will move {direction} with {confidence:.1%} confidence."

    feature_explanations = []
    for feature, value in top_features:
        impact = "toward UP" if value > 0 else "toward DOWN"
        clean_name = feature.replace('_', ' ')

        if 'MA' in feature:
            clean_name = f"{feature.split('_')[-1]}-day moving average"
        elif 'Volatility' in feature:
            clean_name = f"{feature.split('_')[-1]}-day volatility"
        elif 'Volume' in feature and 'MA' in feature:
            clean_name = "3-day average trading volume"
        elif 'High_Low_Pct' in feature:
            clean_name = "daily price range"
        elif 'Returns' in feature:
            clean_name = "daily returns"

        feature_explanations.append(f"{clean_name} pushed prediction {impact} (impact: {value:+.4f})")

    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            reasons = ". ".join(feature_explanations[:2])
            return f"The model predicts {ticker} will move {direction} with {confidence:.1%} confidence. {reasons}."

        response = requests.post(
            "https://router.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"},
            json={"inputs": f"{ticker} {direction} ({confidence:.0%}): {'; '.join(feature_explanations)}",
                  "parameters": {"max_length": 150, "min_length": 30, "do_sample": False}},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get('summary_text') or result[0].get('generated_text', '')

        reasons = ". ".join(feature_explanations[:2])
        return f"The model predicts {ticker} will move {direction} with {confidence:.1%} confidence. {reasons}."

    except Exception:
        reasons = ". ".join(feature_explanations[:2])
        return f"The model predicts {ticker} will move {direction} with {confidence:.1%} confidence. {reasons}."


@app.get("/")
def read_root():
    return {
        "welcome": "Stock Direction Classifier API",
        "model": MODEL_NAME,
        "version": model_version.version
    }


@app.get("/predict_direction")
def predict_direction():
    """Predict stock direction (UP/DOWN) for all tickers."""
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
    probabilities = model.predict_proba(X_latest)

    shap_values_2d = None
    if explainer:
        shap_values_2d = normalize_shap_values(explainer.shap_values(X_latest))

    response = {}
    for i, ticker in enumerate(tickers):
        direction = "UP" if predictions[i] == 1 else "DOWN"
        confidence = probabilities[i][predictions[i]]

        ticker_response = {
            "date": LATEST_DATE.date().isoformat(),
            "direction": direction,
            "confidence": float(confidence),
            "probabilities": {"DOWN": float(probabilities[i][0]), "UP": float(probabilities[i][1])},
            "signal": "BUY" if direction == "UP" and confidence > 0.6 else
                     "SELL" if direction == "DOWN" and confidence > 0.6 else "HOLD"
        }

        if shap_values_2d is not None:
            ticker_shap = {feature_cols[j]: float(shap_values_2d[i][j]) for j in range(len(feature_cols))}
            ticker_response["shap_values"] = ticker_shap
            ticker_response["summary"] = get_llm_summary(ticker_shap, direction, confidence, ticker)

        response[ticker] = ticker_response

    # Save predictions locally
    prediction_df = pd.DataFrame({
        "Date": LATEST_DATE.date(), "Ticker": tickers,
        "Direction_Prediction": predictions,
        "Confidence": [probabilities[i][predictions[i]] for i in range(len(tickers))],
        "Prob_DOWN": probabilities[:, 0], "Prob_UP": probabilities[:, 1]
    })
    prediction_df = pd.concat([prediction_df, X_latest.reset_index(drop=True)], axis=1)

    try:
        existing = pd.read_csv(DIRECTION_PREDICTIONS_PATH)
        prediction_df = pd.concat([existing, prediction_df], ignore_index=True)
    except FileNotFoundError:
        pass
    prediction_df.to_csv(DIRECTION_PREDICTIONS_PATH, index=False)

    # Save SHAP values locally
    if shap_values_2d is not None:
        shap_df = pd.DataFrame(shap_values_2d, columns=feature_cols)
        shap_df.insert(0, "Ticker", tickers)
        shap_df.insert(0, "Date", LATEST_DATE.date())

        try:
            existing_shap = pd.read_csv(DIRECTION_SHAP_PATH)
            shap_df = pd.concat([existing_shap, shap_df], ignore_index=True)
        except FileNotFoundError:
            pass
        shap_df.to_csv(DIRECTION_SHAP_PATH, index=False)

    return response


@app.get("/predict_direction/{ticker}")
def predict_single_ticker(ticker: str):
    """Predict direction for a single ticker."""
    ticker = ticker.upper()

    online_features = store.get_online_features(
        features=FEAST_FEATURES, entity_rows=[{"ticker": ticker}]
    ).to_dict()

    feature_cols = [f.split(':')[-1] for f in FEAST_FEATURES]
    X_latest = pd.DataFrame(online_features).rename(
        columns={f: f.split(':')[-1] for f in FEAST_FEATURES}
    )[feature_cols].astype(float)

    prediction = model.predict(X_latest)[0]
    probabilities = model.predict_proba(X_latest)[0]
    direction = "UP" if prediction == 1 else "DOWN"
    confidence = probabilities[prediction]

    top_features = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:5]

    result = {
        "ticker": ticker,
        "date": LATEST_DATE.date().isoformat(),
        "direction": direction,
        "confidence": float(confidence),
        "probabilities": {"DOWN": float(probabilities[0]), "UP": float(probabilities[1])},
        "signal": "BUY" if direction == "UP" and confidence > 0.6 else
                 "SELL" if direction == "DOWN" and confidence > 0.6 else "HOLD",
        "top_features": [{"feature": f, "importance": float(i)} for f, i in top_features]
    }

    if explainer:
        shap_values = normalize_shap_values(explainer.shap_values(X_latest))
        ticker_shap = {feature_cols[j]: float(shap_values[0][j]) for j in range(len(feature_cols))}
        result["shap_values"] = ticker_shap
        result["summary"] = get_llm_summary(ticker_shap, direction, confidence, ticker)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
