"""Direction Classifier Model Monitoring using Evidently AI."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from config.global_config import FEATURES_LIST, PROCESSED_DATA_PATH, PREDICTIONS_PATH, DRIFT_REPORT_PATH
from .config import MODEL_NAME

FEATURES = FEATURES_LIST
DIRECTION_PREDICTIONS_PATH = PREDICTIONS_PATH.parent / "direction_predictions.csv"
DIRECTION_DRIFT_REPORT_PATH = DRIFT_REPORT_PATH.parent / f"{MODEL_NAME}_data_drift_report.html"
CLASSIFICATION_REPORT_PATH = DRIFT_REPORT_PATH.parent / f"{MODEL_NAME}_classification_report.html"


def build_dataset(df, features):
    data_def = DataDefinition(numerical_columns=features, categorical_columns=[])
    return Dataset.from_pandas(df[features], data_definition=data_def)


def main():
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} Monitoring - {datetime.now():%Y-%m-%d %H:%M}")
    print('='*50)

    if not PROCESSED_DATA_PATH.exists():
        print("No training data. Run feature_engineering.py first.")
        return

    if not DIRECTION_PREDICTIONS_PATH.exists():
        print("No predictions. Run serve.py first.")
        return

    ref_df = pd.read_csv(PROCESSED_DATA_PATH).dropna(subset=FEATURES)
    cur_df = pd.read_csv(DIRECTION_PREDICTIONS_PATH)
    print(f"Training: {len(ref_df)}, Predictions: {len(cur_df)}")

    # Drift report
    features = [c for c in FEATURES if c in cur_df.columns]
    report = Report([DataDriftPreset()])
    report.run(
        current_data=build_dataset(cur_df.dropna(subset=features), features),
        reference_data=build_dataset(ref_df, features)
    ).save_html(str(DIRECTION_DRIFT_REPORT_PATH))
    print(f"Drift report: {DIRECTION_DRIFT_REPORT_PATH}")

    # Classification report
    if "Direction_Actual" not in cur_df.columns or cur_df["Direction_Actual"].isna().all():
        print("No actuals - run update_actuals.py first")
        return

    with_actuals = cur_df.dropna(subset=["Direction_Prediction", "Direction_Actual"])
    accuracy = (with_actuals["Direction_Prediction"] == with_actuals["Direction_Actual"]).mean()
    print(f"Accuracy: {accuracy:.2%} ({len(with_actuals)} samples)")

    report = Report([DataDriftPreset()])
    report.run(
        current_data=build_dataset(with_actuals, features),
        reference_data=build_dataset(ref_df, features)
    ).save_html(str(CLASSIFICATION_REPORT_PATH))
    print(f"Classification report: {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
