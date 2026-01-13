import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from config import (
    FEATURES_LIST, PROCESSED_DATA_KEY, PREDICTIONS_ACTUALS_KEY,
    REPORTS_DIR, DRIFT_REPORT_KEY, REGRESSION_REPORT_KEY
)

from storage import MinioArtifactStore
store = MinioArtifactStore()

TRAINING_DATA = PROCESSED_DATA_KEY
PREDICTIONS_DATA = PREDICTIONS_ACTUALS_KEY
FEATURES = FEATURES_LIST

def load_training_data():
    try:
        df = store.load_df(TRAINING_DATA).dropna(subset=FEATURES)
        print(f"Loaded {len(df)} training samples from MinIO")
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise

def load_predictions():
    try:
        df = store.load_df(PREDICTIONS_DATA)
    except Exception:
        print("Predictions file not found in MinIO.")
        return None, None

    cols = [c for c in FEATURES if c in df.columns]

    all_preds = df.dropna(subset=cols)
    with_actuals = df.dropna(subset=['Actual'] + cols)

    print(f"Loaded {len(all_preds)} predictions ({len(with_actuals)} with actuals)")
    return all_preds, with_actuals if len(with_actuals) > 0 else None


def build_dataset(df, features, regression_cols=None):
    data_def = DataDefinition(
        numerical_columns=features,
        categorical_columns=[],
        regression=[Regression(target="Actual", prediction="Prediction")] if regression_cols else None
    )
    cols = features + (regression_cols or [])
    return Dataset.from_pandas(df[cols], data_definition=data_def)


def drift_report(ref_df, cur_df):
    features = [c for c in FEATURES if c in ref_df.columns and c in cur_df.columns]
    print(f"Comparing {len(features)} features for drift")

    report = Report([DataDriftPreset()])
    result = report.run(
        current_data=build_dataset(cur_df, features),
        reference_data=build_dataset(ref_df, features)
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "data_drift_report.html"
    result.save_html(str(path))

    store.upload_file(str(path), DRIFT_REPORT_KEY)

    return path


def regression_report(ref_df, cur_df):
    if 'Prediction' not in cur_df.columns or 'Actual' not in cur_df.columns:
        return None
    if 'Target' not in ref_df.columns:
        return None

    features = [c for c in FEATURES if c in ref_df.columns and c in cur_df.columns]

    ref = ref_df.copy()
    ref['Prediction'] = ref['Target']
    ref['Actual'] = ref['Target']
    ref = ref.dropna(subset=['Prediction', 'Actual'] + features)
    cur = cur_df.dropna(subset=['Prediction', 'Actual'] + features)

    if len(cur) == 0:
        return None

    report = Report([RegressionPreset()])
    result = report.run(
        current_data=build_dataset(cur, features, ['Prediction', 'Actual']),
        reference_data=build_dataset(ref, features, ['Prediction', 'Actual'])
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "regression_performance_report.html"
    result.save_html(str(path))

    store.upload_file(str(path), REGRESSION_REPORT_KEY)

    return path


def main():
    print(f"\n{'='*50}")
    print(f"Evidently Monitoring - {datetime.now():%Y-%m-%d %H:%M}")
    print('='*50)
    
    try:
        ref_df = load_training_data()
        cur_df, actuals_df = load_predictions()
        
        if cur_df is not None:
            drift_path = drift_report(ref_df, cur_df)
            print(f"  - data_drift_report.html")
        else:
            print("  - (no drift report - missing predictions)")
            
        reg_path = regression_report(ref_df, actuals_df) if actuals_df is not None else None
        print(f"  - regression_performance_report.html" if reg_path else "  - (no regression report - missing actuals)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
