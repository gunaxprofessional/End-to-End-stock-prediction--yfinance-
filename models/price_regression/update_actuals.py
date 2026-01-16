"""Update price predictions with actual closing values."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from config.global_config import PREDICTIONS_PATH, RAW_DATA_PATH


def update_actuals():
    print("=" * 60)
    print("PRICE REGRESSION - UPDATING ACTUAL VALUES")
    print("=" * 60)

    if not PREDICTIONS_PATH.exists():
        print("No predictions file. Run serve.py first.")
        return

    if not RAW_DATA_PATH.exists():
        print("No stock data. Run data_ingestion.py first.")
        return

    predictions = pd.read_csv(PREDICTIONS_PATH)
    stock_data = pd.read_csv(RAW_DATA_PATH)
    print(f"Predictions: {len(predictions)}, Stock data: {len(stock_data)}")

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    if "Actual" not in predictions.columns:
        predictions["Actual"] = None

    missing = predictions["Actual"].isna().sum()
    if missing == 0:
        print("All rows already have actuals.")
        return

    # Actual: next day's Close price
    stock_sorted = stock_data.sort_values(["Ticker", "Date"]).copy()
    stock_sorted["PredDate"] = stock_sorted.groupby("Ticker")["Date"].shift(1)

    lookup = stock_sorted[["Ticker", "PredDate", "Close"]].dropna()
    lookup = lookup.rename(columns={"Close": "ActualClose"})

    merged = predictions.merge(lookup, left_on=["Ticker", "Date"], right_on=["Ticker", "PredDate"], how="left")

    mask = predictions["Actual"].isna() & merged["ActualClose"].notna()
    predictions.loc[mask, "Actual"] = merged.loc[mask, "ActualClose"]
    predictions = predictions.drop(columns=["PredDate", "ActualClose"], errors="ignore")

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    updated = missing - predictions["Actual"].isna().sum()
    print(f"Updated: {updated}, Saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    update_actuals()
