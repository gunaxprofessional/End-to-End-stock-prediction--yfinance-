import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import datetime

from feast import FeatureStore

from config.global_config import (
    FEAST_FEATURES, TRACKING_URI, PROCESSED_DATA_PATH,
    TEAM_NAME, PROJECT_NAME, FEAST_REPO_PATH
)
from models.direction_classifier.config import (
    MODEL_NAME, EXPERIMENT_NAME, MODEL_PARAMS,
    TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE,
    MODEL_ALIAS, TARGET
)

store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
features = FEAST_FEATURES
mlflow.set_tracking_uri(str(TRACKING_URI))


def create_target_variable(data):
    """Create binary target: 1 if price goes up, 0 if down."""
    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    data['Next_Close'] = data.groupby('Ticker')['Close'].shift(-1)
    data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)
    data = data.dropna(subset=['Direction'])
    print(f"Direction distribution:\n{data['Direction'].value_counts()}")
    return data


def load_data_store():
    """Load data from Feast feature store."""
    print("Loading data from Feast feature store...")
    entity_df = pd.read_csv(PROCESSED_DATA_PATH)
    entity_df['event_timestamp'] = pd.to_datetime(entity_df['Date'], utc=True)
    entity_df['ticker'] = entity_df['Ticker']
    print(f"Entity DataFrame loaded. Total records: {len(entity_df)}")

    # Filter to date range
    filter_start = pd.to_datetime(TRAIN_START_DATE, utc=True) - pd.Timedelta(days=1)
    filter_end = pd.to_datetime(TEST_END_DATE, utc=True) + pd.Timedelta(days=1)

    entity_df_filtered = entity_df[
        (entity_df['event_timestamp'] >= filter_start) &
        (entity_df['event_timestamp'] <= filter_end)
    ].copy()

    print(f"Filtered to date range: {len(entity_df_filtered)} records")

    print("Retrieving historical features from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df_filtered[['ticker', 'event_timestamp']],
        features=features
    ).to_df()

    print(f"Historical features retrieved. Records: {len(training_df)}")

    training_df = training_df.merge(
        entity_df_filtered[['ticker', 'event_timestamp', 'Date', 'Ticker']],
        on=['ticker', 'event_timestamp']
    )

    training_df = create_target_variable(training_df)
    return training_df


def prepare_data(data):
    """Prepare training and test sets."""
    train_data = data[
        (data['Date'] >= TRAIN_START_DATE) & (data['Date'] <= TRAIN_END_DATE)
    ]
    test_data = data[
        (data['Date'] >= TEST_START_DATE) & (data['Date'] <= TEST_END_DATE)
    ]

    feature_cols = [f.split(':')[-1] for f in features]

    X_train = train_data[feature_cols]
    y_train = train_data[TARGET]
    X_test = test_data[feature_cols]
    y_test = test_data[TARGET]

    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    return X_train, y_train, X_test, y_test, feature_cols


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"Metrics: {metrics}")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    return metrics


def main():
    print("=" * 60)
    print("Stock Direction Classifier - Training Pipeline")
    print("=" * 60)

    data = load_data_store()
    X_train, y_train, X_test, y_test, feature_cols = prepare_data(data)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"direction_classifier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params(MODEL_PARAMS)
        mlflow.set_tags({"team": TEAM_NAME, "project": PROJECT_NAME})

        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)

        print(f"Model registered: {MODEL_NAME}, Version: {registered_model.version}")

        # Champion challenge
        champion_f1 = 0
        try:
            champion_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            champion_run = client.get_run(champion_version.run_id)
            champion_f1 = champion_run.data.metrics.get('f1_score', 0)
        except Exception:
            pass

        if metrics['f1_score'] > champion_f1:
            client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, registered_model.version)
            print(f"Version {registered_model.version} promoted to {MODEL_ALIAS}")

    print("Training complete!")


if __name__ == "__main__":
    main()
