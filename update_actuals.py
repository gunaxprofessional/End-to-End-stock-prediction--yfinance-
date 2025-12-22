import pandas as pd
from pathlib import Path

"""
- A prediction made on Date X predicts the closing price for Date X+1
- The Actual value = Close price from the next trading day after Date X
"""

# Paths
PREDICTIONS_PATH = Path("data") / "predictions" / "predictions.csv"
STOCK_DATA_PATH = Path("data") / "raw" / "stock_data.csv"
OUTPUT_PATH = Path("data") / "predictions" / "predictions_with_actuals.csv"


def update_actuals():
    """Update predictions with actual closing prices using vectorized merge."""
    
    print("=" * 60)
    print("UPDATING ACTUAL VALUES IN PREDICTIONS")
    print("=" * 60)
    
    # Check if predictions file exists
    if not PREDICTIONS_PATH.exists():
        print("No predictions file found. Nothing to update.")
        return
    
    # Load data
    print("\nLoading predictions...")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    print(f"Total prediction rows: {len(predictions)}")
    
    print("Loading stock data...")
    stock_data = pd.read_csv(STOCK_DATA_PATH)
    print(f"Stock data rows: {len(stock_data)}")
    
    # Convert dates
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    
    # Count rows needing update before
    missing_before = predictions["Actual"].isna().sum()
    print(f"\nRows missing Actual values: {missing_before}")
    
    if missing_before == 0:
        print("All rows already have Actual values. Nothing to update.")
        return
        
    stock_sorted = stock_data.sort_values(["Ticker", "Date"]).copy()
    stock_sorted["PredDate"] = stock_sorted.groupby("Ticker")["Date"].shift(1)
    
    # Keep only what we need for merging
    lookup = stock_sorted[["Ticker", "PredDate", "Close"]].dropna(subset=["PredDate"])
    lookup = lookup.rename(columns={"Close": "ActualClose"})
    
    # Merge predictions with lookup on (Ticker, Date == PredDate)
    merged = predictions.merge(
        lookup,
        left_on=["Ticker", "Date"],
        right_on=["Ticker", "PredDate"],
        how="left"
    )
    
    # Fill Actual only where it was missing
    mask = predictions["Actual"].isna() & merged["ActualClose"].notna()
    predictions.loc[mask, "Actual"] = merged.loc[mask, "ActualClose"]
    
    # Cleanup
    predictions = predictions.drop(columns=["PredDate", "ActualClose"], errors="ignore")
    
    # Save updated predictions
    predictions.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    missing_after = predictions["Actual"].isna().sum()
    updated_count = missing_before - missing_after
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Updated rows: {updated_count}")
    print(f"Still missing: {missing_after}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    update_actuals()
