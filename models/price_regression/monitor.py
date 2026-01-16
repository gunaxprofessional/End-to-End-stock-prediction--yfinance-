"""Price Regression Model Monitoring using Evidently AI."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datetime import datetime
import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from config.global_config import FEATURES_LIST, PROCESSED_DATA_PATH, PREDICTIONS_PATH, DRIFT_REPORT_PATH, REGRESSION_REPORT_PATH
from .config import MODEL_NAME

FEATURES = FEATURES_LIST


def build_dataset(df, features, regression_cols=None):
    data_def = DataDefinition(
        numerical_columns=features,
        categorical_columns=[],
        regression=[Regression(target="Actual", prediction="Prediction")] if regression_cols else None
    )
    return Dataset.from_pandas(df[features + (regression_cols or [])], data_definition=data_def)


def main():
    print(f"\n{'='*50}")
    print(f"{MODEL_NAME} Monitoring - {datetime.now():%Y-%m-%d %H:%M}")
    print('='*50)

    if not PROCESSED_DATA_PATH.exists():
        print("No training data. Run feature_engineering.py first.")
        return

    if not PREDICTIONS_PATH.exists():
        print("No predictions. Run serve.py first.")
        return

    ref_df = pd.read_csv(PROCESSED_DATA_PATH).dropna(subset=FEATURES)
    cur_df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Training: {len(ref_df)}, Predictions: {len(cur_df)}")

    # Drift report
    features = [c for c in FEATURES if c in cur_df.columns]
    report = Report([DataDriftPreset()])
    report.run(
        current_data=build_dataset(cur_df.dropna(subset=features), features),
        reference_data=build_dataset(ref_df, features)
    ).save_html(str(DRIFT_REPORT_PATH))
    print(f"Drift report: {DRIFT_REPORT_PATH}")

    # Regression report
    if "Actual" not in cur_df.columns or cur_df["Actual"].isna().all():
        print("No actuals - run update_actuals.py first")
        return

    with_actuals = cur_df.dropna(subset=["Prediction", "Actual"])
    ref = ref_df.copy()
    ref["Prediction"], ref["Actual"] = ref["Target"], ref["Target"]
    ref = ref.dropna(subset=["Prediction", "Actual"] + features)

    report = Report([RegressionPreset()])
    report.run(
        current_data=build_dataset(with_actuals, features, ["Prediction", "Actual"]),
        reference_data=build_dataset(ref, features, ["Prediction", "Actual"])
    ).save_html(str(REGRESSION_REPORT_PATH))
    print(f"Regression report: {REGRESSION_REPORT_PATH}")


if __name__ == "__main__":
    main()
