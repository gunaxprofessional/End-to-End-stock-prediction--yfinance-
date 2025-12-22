import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuration
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# END DATE
END_DATE = datetime.now().date() - timedelta(days=1)  # yesterday

END_DATE = pd.to_datetime(END_DATE)

# START DATE
START_DATE = pd.to_datetime("22-12-2024",format="%d-%m-%Y")

print("=" * 60)
print("STOCK PRICE PREDICTION - NEXT DAY CLOSING")
print("=" * 60)
print(f"\nFetching data from {START_DATE} to {END_DATE.date()}")
print(f"Stocks: {', '.join(STOCKS)}\n")

# Fetch stock data
all_data = []
for ticker in STOCKS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Ticker'] = ticker
    df.reset_index(inplace=True)
    all_data.append(df)

# Combine all stock data
data = pd.concat(all_data, ignore_index=True)

print(f"\nTotal records fetched: {len(data)}")
print(
    f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")

print("\nSaving raw data to 'stock_data.csv'...")
output_path = Path("data") / "raw" / "stock_data.csv"
data.to_csv(output_path, index=False)
