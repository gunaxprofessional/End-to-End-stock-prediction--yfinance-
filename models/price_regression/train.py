import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import datetime

from feast import FeatureStore

from config.global_config import (
    FEAST_FEATURES, TRACKING_URI, PROCESSED_DATA_PATH,
    TEAM_NAME, PROJECT_NAME, FEAST_REPO_PATH
)
from models.price_regression.config import (
    MODEL_NAME, EXPERIMENT_NAME,
    TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE,
    MODEL_ALIAS
)

store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
features = FEAST_FEATURES
mlflow.set_tracking_uri(str(TRACKING_URI))


def load_data_store():
    """Load data from Feast feature store."""
    print("Loading data from Feast feature store...")
    entity_df = pd.read_csv(PROCESSED_DATA_PATH)
    entity_df['event_timestamp'] = pd.to_datetime(entity_df['Date'], utc=True)
    entity_df['ticker'] = entity_df['Ticker']
    print(f"Entity DataFrame loaded. Total records: {len(entity_df)}")

    # Filter to date range
    filter_start = pd.to_datetime(TRAIN_START_DATE, utc=True)
    filter_end = pd.to_datetime(TEST_END_DATE, utc=True)

    entity_df_filtered = entity_df[
        (entity_df['event_timestamp'] >= filter_start) &
        (entity_df['event_timestamp'] <= filter_end)
    ].copy()

    print(f"Filtered to date range: {len(entity_df_filtered)} records")

    print("Retrieving historical features from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df_filtered[['event_timestamp', 'ticker', 'Target']],
        features=features
    ).to_df()

    training_df.dropna(inplace=True)

    print(f"Feast retrieval complete. Records: {len(training_df)}")

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
        raise ValueError("Train or test data is empty.")

    print(f"Training data records: {len(train_data)}")
    print(f"Testing data records: {len(test_data)}")

    return train_data, test_data


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return rmse, mae, r2


def model_training(train_data, test_data):
    print("Starting model training...")
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_cols = [col for col in train_data.columns if col.lower() not in ['date', 'ticker', 'target']]
    X_train, y_train = train_data[feature_cols], train_data['Target']
    X_test, y_test = test_data[feature_cols], test_data['Target']

    with mlflow.start_run(run_name=f"RF_Train_{datetime.date.today()}") as run:
        mlflow.set_tags({"team": TEAM_NAME, "project": PROJECT_NAME})

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse, mae, r2 = calculate_metrics(y_test, y_pred)

        signature = infer_signature(X_test.astype(float), y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="stock_model_artifact",
            signature=signature,
            input_example=X_test.iloc[:3].astype(float),
            registered_model_name=MODEL_NAME
        )

        mlflow.log_params({"n_estimators": 100, "features_count": len(feature_cols)})
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})

        version = model_info.registered_model_version

        # Champion challenge
        champion_r2 = -1.0
        try:
            champion_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            champion_run = client.get_run(champion_version.run_id)
            champion_r2 = champion_run.data.metrics.get("R2", 0.0)
        except Exception:
            pass

        if r2 > champion_r2:
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, version)
            print(f"Version {version} promoted to {MODEL_ALIAS}")

        return run.info.run_id


if __name__ == "__main__":
    print("Pipeline started...")
    data = load_data_store()
    train, test = split_data(data)
    run_id = model_training(train, test)
    print(f"Pipeline finished. Run ID: {run_id}")
