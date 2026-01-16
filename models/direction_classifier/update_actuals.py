"""Update direction predictions with actual direction values."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from config.global_config import PREDICTIONS_PATH, RAW_DATA_PATH

DIRECTION_PREDICTIONS_PATH = PREDICTIONS_PATH.parent / "direction_predictions.csv"


def update_actuals():
    print("=" * 60)
    print("DIRECTION CLASSIFIER - UPDATING ACTUAL VALUES")
    print("=" * 60)

    if not DIRECTION_PREDICTIONS_PATH.exists():
        print("No predictions file. Run serve.py first.")
        return

    if not RAW_DATA_PATH.exists():
        print("No stock data. Run data_ingestion.py first.")
        return

    predictions = pd.read_csv(DIRECTION_PREDICTIONS_PATH)
    stock_data = pd.read_csv(RAW_DATA_PATH)
    print(f"Predictions: {len(predictions)}, Stock data: {len(stock_data)}")

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    if "Direction_Actual" not in predictions.columns:
        predictions["Direction_Actual"] = None

    missing = predictions["Direction_Actual"].isna().sum()
    if missing == 0:
        print("All rows already have actuals.")
        return

    # Actual direction: 1 if next day Close > current Close, else 0
    stock_sorted = stock_data.sort_values(["Ticker", "Date"]).copy()
    stock_sorted["NextClose"] = stock_sorted.groupby("Ticker")["Close"].shift(-1)
    stock_sorted["ActualDirection"] = (stock_sorted["NextClose"] > stock_sorted["Close"]).astype(int)

    lookup = stock_sorted[["Ticker", "Date", "ActualDirection"]].dropna()
    merged = predictions.merge(lookup, on=["Ticker", "Date"], how="left")

    mask = predictions["Direction_Actual"].isna() & merged["ActualDirection"].notna()
    predictions.loc[mask, "Direction_Actual"] = merged.loc[mask, "ActualDirection"].astype(int)

    predictions.to_csv(DIRECTION_PREDICTIONS_PATH, index=False)

    updated = missing - predictions["Direction_Actual"].isna().sum()
    with_actuals = predictions.dropna(subset=["Direction_Actual"])
    if len(with_actuals) > 0:
        accuracy = (with_actuals["Direction_Prediction"] == with_actuals["Direction_Actual"]).mean()
        print(f"Accuracy: {accuracy:.2%}")

    print(f"Updated: {updated}, Saved to: {DIRECTION_PREDICTIONS_PATH}")


if __name__ == "__main__":
    update_actuals()
