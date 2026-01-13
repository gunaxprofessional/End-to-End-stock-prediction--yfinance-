import pandas as pd
from pathlib import Path

from config import PREDICTIONS_KEY, RAW_DATA_KEY, PREDICTIONS_ACTUALS_KEY
from storage import MinioArtifactStore
store = MinioArtifactStore()

def update_actuals():
    print("=" * 60)
    print("UPDATING ACTUAL VALUES IN PREDICTIONS")
    print("=" * 60)

    print("\nLoading predictions from MinIO...")
    try:
        predictions = store.load_df(PREDICTIONS_KEY)
        print(f"Total prediction rows: {len(predictions)}")
    except Exception:
        print("No predictions file found in MinIO. Nothing to update.")
        return

    print("Loading stock data from MinIO...")
    try:
        stock_data = store.load_df(RAW_DATA_KEY)
        print(f"Stock data rows: {len(stock_data)}")
    except Exception:
        print("No stock data file found in MinIO. Cannot update actuals.")
        return

    predictions["Date"] = pd.to_datetime(predictions["Date"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])

    if "Actual" not in predictions.columns:
        predictions["Actual"] = None

    missing_before = predictions["Actual"].isna().sum()
    print(f"\nRows missing Actual values: {missing_before}")

    if missing_before == 0:
        print("All rows already have Actual values. Nothing to update.")
        return

    stock_sorted = stock_data.sort_values(["Ticker", "Date"]).copy()
    stock_sorted["PredDate"] = stock_sorted.groupby("Ticker")["Date"].shift(1)

    lookup = stock_sorted[["Ticker", "PredDate", "Close"]].dropna(subset=["PredDate"])
    lookup = lookup.rename(columns={"Close": "ActualClose"})

    merged = predictions.merge(
        lookup,
        left_on=["Ticker", "Date"],
        right_on=["Ticker", "PredDate"],
        how="left"
    )

    mask = predictions["Actual"].isna() & merged["ActualClose"].notna()
    predictions.loc[mask, "Actual"] = merged.loc[mask, "ActualClose"]

    predictions = predictions.drop(columns=["PredDate", "ActualClose"], errors="ignore")

    store.save_df(predictions, PREDICTIONS_ACTUALS_KEY)

    missing_after = predictions["Actual"].isna().sum()
    updated_count = missing_before - missing_after

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Updated rows: {updated_count}")
    print(f"Still missing: {missing_after}")
    print(f"Saved to: data/predictions/predictions_with_actuals.csv in MinIO")


if __name__ == "__main__":
    update_actuals()
