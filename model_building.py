import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime

from feast import FeatureStore
from storage import MinioArtifactStore

from config import (
    MODEL_NAME, EXPERIMENT_NAME, FEAST_FEATURES, TRACKING_URI,
    PROCESSED_DATA_KEY, FEATURE_PARQUET_KEY,
    TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE,
    TEAM_NAME, PROJECT_NAME
)

store = FeatureStore(repo_path="feature_repo")
minio_store = MinioArtifactStore()

features = FEAST_FEATURES
mlflow.set_tracking_uri(TRACKING_URI)

def load_data():
    print("Loading data from MinIO...")
    try:
        data = minio_store.load_df(PROCESSED_DATA_KEY)
        print(f"Data loaded successfully. Total records: {len(data)}")
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def load_data_store():
    print("Loading data from Feast feature store...")
    try:
        entity_df = minio_store.load_df(PROCESSED_DATA_KEY)
        entity_df['event_timestamp'] = pd.to_datetime(entity_df['Date'], utc=True)
        entity_df['ticker'] = entity_df['Ticker']
        print(f"Entity DataFrame loaded. Records: {len(entity_df)}")
    except Exception as e:
        raise Exception(f"Error loading data from Feast: {str(e)}")

    print("Retrieving historical features from Feast...")
    minio_store.download_file(FEATURE_PARQUET_KEY, FEATURE_PARQUET_KEY)

    training_df = store.get_historical_features(
        entity_df=entity_df[['event_timestamp', 'ticker', 'Target']],
        features=features
    ).to_df()

    training_df.dropna(inplace=True)

    print(f"Feast retrieval complete. Records: {len(training_df)}")
    print(f"Training DF Columns: {training_df.columns.tolist()}")

    training_df = training_df.rename(columns={"event_timestamp": "Date"})
    training_df['Date'] = pd.to_datetime(training_df['Date']).dt.tz_localize(None)

    return training_df

def split_data(data):
    train_start_date = pd.to_datetime(TRAIN_START_DATE)
    train_end_date = pd.to_datetime(TRAIN_END_DATE)
    test_start_date = pd.to_datetime(TEST_START_DATE)
    test_end_date = pd.to_datetime(TEST_END_DATE)

    data['Date'] = pd.to_datetime(data['Date'])

    train_data = data[(data['Date'] >= train_start_date) & (data['Date'] <= train_end_date)].copy()
    test_data = data[(data['Date'] >= test_start_date) & (data['Date'] <= test_end_date)].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train or test data is empty. Please check the date ranges and data availability.")

    print(f"Training data records: {len(train_data)}")
    print(f"Testing data records: {len(test_data)}")

    return train_data, test_data


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return rmse, mae, r2

def model_training(train_data, test_data):
    print("Starting model training...")
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_cols = [col for col in train_data.columns if col.lower() not in ['date', 'ticker', 'target', 'event_timestamp']]
    print(f"Feature columns: {feature_cols}")
    X_train, y_train = train_data[feature_cols], train_data['Target']
    X_test, y_test = test_data[feature_cols], test_data['Target']

    tags = {
        "team": TEAM_NAME,
        "project": PROJECT_NAME,
        "priority": "High",
        "created_by": os.getenv("USER", "CI_CD_Pipeline"),
        "framework": "scikit-learn"
    }

    with mlflow.start_run(run_name=f"RF_Train_{datetime.date.today()}") as run:
        print("MLflow run started...")
        mlflow.set_tags(tags)

        print("Training RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model training completed.")

        y_pred = model.predict(X_test)
        print("Predictions on test set completed.")

        rmse, mae, r2 = calculate_metrics(y_test, y_pred)

        signature = infer_signature(X_test.astype(float), y_pred)

        print("Logging model to MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="stock_model_artifact",
            signature=signature,
            input_example=X_test.iloc[:3].astype(float),
            registered_model_name=MODEL_NAME
        )
        print("Model logged successfully.")

        version = model_info.registered_model_version

        print("Updating registered model description...")
        client.update_registered_model(
            name=MODEL_NAME,
            description="Production model for predicting daily stock price movements."
        )

        print("Updating model version description...")
        client.update_model_version(
            name=MODEL_NAME,
            version=version,
            description=f"Automated Training. R2: {r2:.4f}. Dataset size: {len(train_data)} rows."
        )

        print("Logging metrics and parameters...")
        mlflow.log_params({"n_estimators": 100, "features_count": len(feature_cols)})
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})

        print("--- Challenge the Champion ---")
        champion_r2 = -1.0
        try:
            champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champion_run_id = champion_version.run_id
            champion_run = client.get_run(champion_run_id)
            champion_r2 = champion_run.data.metrics.get("R2", 0.0)
            print(f"Current Champion Version: {champion_version.version}, R2: {champion_r2:.4f}")
        except Exception:
            print("No existing champion found. Initializing baseline.")

        if r2 > 0.85 and r2 > champion_r2:
            print(f"New Model R2 ({r2:.4f}) > Champion R2 ({champion_r2:.4f}). Promoting...")
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            print(f"Version {version} is now the CHAMPION.")
        else:
            print(f"New Model R2 ({r2:.4f}) <= Champion R2 ({champion_r2:.4f}) or Threshold (0.85).")
            print(f"Version {version} remains a candidate.")

        print("Model training and logging completed.")

        try:
            print("Logging SHAP background dataset...")
            background = X_train.sample(n=min(100, len(X_train)), random_state=42)
            background_path = "shap_background.parquet"
            background.to_parquet(background_path)
            mlflow.log_artifact(background_path, artifact_path="shap")
        except Exception as e:
            print(f"Failed to log SHAP background dataset: {e}")

        return run.info.run_id

if __name__ == "__main__":
    try:
        print("Pipeline started...")
        raw_data_store = load_data_store()
        train, test = split_data(raw_data_store)
        run_id = model_training(train, test)
        print(f"Pipeline finished successfully. Run ID: {run_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline failed: {e}")