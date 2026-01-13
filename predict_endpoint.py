import fastapi
import pandas as pd
import requests
import mlflow
import mlflow.sklearn
import shap
from mlflow.tracking import MlflowClient
from datetime import timedelta, datetime
from pathlib import Path
import os
from feast import FeatureStore
from dotenv import load_dotenv
from storage import MinioArtifactStore
from config import (
    MODEL_NAME, MODEL_ALIAS, TRACKING_URI, FEAST_FEATURES,
    REGISTRY_KEY, ONLINE_STORE_KEY, PROCESSED_DATA_KEY,
    PREDICTIONS_KEY, SHAP_VALUES_KEY, PREDICTION_DATE
)

load_dotenv()

minio_store = MinioArtifactStore()

print("Downloading Feast registry and online store...")
minio_store.download_file(REGISTRY_KEY, "data/registry.db")
minio_store.download_file(ONLINE_STORE_KEY, "data/online_store.db")

store = FeatureStore(repo_path="feature_repo")

LATEST_DATE = pd.to_datetime(PREDICTION_DATE, format="%Y-%m-%d")

app = fastapi.FastAPI()
mlflow.set_tracking_uri(TRACKING_URI)

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
model = mlflow.sklearn.load_model(model_uri)

client = MlflowClient()
model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
run_id = model_version.run_id
background_path = client.download_artifacts(run_id, "shap/shap_background.parquet")
shap_background = pd.read_parquet(background_path)

explainer = shap.TreeExplainer(model, shap_background)

def get_llm_summary(shap_dict: dict, prediction: float, ticker: str) -> str:
    sorted_features = sorted(
        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(feat, val)
                    for feat, val in sorted_features if abs(val) > 0.0001][:5]

    # If no significant SHAP values
    if not top_features:
        return f"The model predicts the next closing price for {ticker} to be {prediction:.2f}. No specific feature showed a significant impact on this prediction."

    # Create detailed explanations
    feature_explanations = []
    for feature, value in top_features:
        direction = "up" if value > 0 else "down"

        # Map to human readable names
        clean_name = feature.replace('_', ' ')
        if 'MA' in feature:
            period = feature.split('_')[-1]
            clean_name = f"{period}-day moving average"
        elif 'Volatility' in feature:
            period = feature.split('_')[-1]
            clean_name = f"{period}-day price volatility"
        elif 'Volume' in feature and 'MA' in feature:
             clean_name = "3-day average trading volume"
        elif 'High_Low_Pct' in feature:
            clean_name = "daily price range (High-Low)"

        feature_explanations.append(
            f"{clean_name} pushed the prediction {direction} (impact: {value:+.4f})")

    prompt = f"""Explain this stock price prediction clearly and concisely:

                Ticker: {ticker}
                Predicted Next Close: {prediction:.2f}

                Key technical factors driving this prediction:
                {chr(10).join(feature_explanations)}

                Write a brief 2-3 sentence explanation for why this prediction was made based on these technical indicators."""

    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            reasons = ". ".join(feature_explanations[:2])
            return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."

        response = requests.post(
            "https://router.huggingface.co/models/facebook/bart-large-cnn",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 150,
                    "min_length": 30,
                    "do_sample": False
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summary = result[0].get('summary_text') or result[0].get(
                    'generated_text', '')
                if summary:
                    return summary
            elif isinstance(result, dict):
                summary = result.get('summary_text') or result.get(
                    'generated_text', '')
                if summary:
                    return summary

        reasons = ". ".join(feature_explanations[:2])
        return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."

    except Exception as e:
        reasons = ". ".join(feature_explanations[:2])
        return f"The predicted close for {ticker} is {prediction:.2f}. {reasons}."

@app.get("/")
def read_root():
    return {"welcome": "welcome to stock price prediction api"}

@app.get("/predict_next_close")
def predict_next_close():
    data = minio_store.load_df(PROCESSED_DATA_KEY)
    data["Date"] = pd.to_datetime(data["Date"])

    tickers = data["Ticker"].unique().tolist()

    print(f"Retrieving online features for {len(tickers)} tickers from Feast...")
    entity_rows = [{"ticker": ticker} for ticker in tickers]

    online_features = store.get_online_features(
        features=FEAST_FEATURES,
        entity_rows=entity_rows
    ).to_dict()

    X_latest_df = pd.DataFrame(online_features)

    feature_map = {f: f.split(':')[-1] for f in FEAST_FEATURES}
    X_latest_df = X_latest_df.rename(columns=feature_map)

    feature_cols = [f.split(':')[-1] for f in FEAST_FEATURES]
    X_latest = X_latest_df[feature_cols].astype(float)

    predictions = model.predict(X_latest)
    shap_values = explainer.shap_values(X_latest)

    response = {}
    for i, ticker in enumerate(tickers):
        ticker_shap = {
            feature_cols[j]: float(shap_values[i][j])
            for j in range(len(feature_cols))
        }

        summary = get_llm_summary(ticker_shap, float(predictions[i]), ticker)

        response[ticker] = {
            "date": LATEST_DATE.date(),
            "prediction": float(predictions[i]),
            "summary": summary,
            "shap_values": ticker_shap
        }

    prediction_df = pd.DataFrame({
        "Date": LATEST_DATE.date(),
        "Ticker": tickers,
        "Prediction": predictions,
        "Actual": None
    })

    prediction_df = pd.concat([prediction_df, X_latest.reset_index(drop=True)], axis=1)

    try:
        existing_df = minio_store.load_df(PREDICTIONS_KEY)
        prediction_df = pd.concat([existing_df, prediction_df], ignore_index=True)
    except:
        pass

    minio_store.save_df(prediction_df, PREDICTIONS_KEY)

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df.insert(0, "Ticker", tickers)
    shap_df.insert(0, "Date", LATEST_DATE.date())

    try:
        existing_shap = minio_store.load_df(SHAP_VALUES_KEY)
        shap_df = pd.concat([existing_shap, shap_df], ignore_index=True)
    except:
        pass

    minio_store.save_df(shap_df, SHAP_VALUES_KEY)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
