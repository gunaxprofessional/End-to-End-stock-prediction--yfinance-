import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.global_config import STOCKS, START_DATE, RAW_DATA_PATH

END_DATE = pd.to_datetime(datetime.now().date() - timedelta(days=1))
START_DATE = pd.to_datetime(START_DATE)

print("=" * 60)
print("STOCK PRICE PREDICTION - NEXT DAY CLOSING")
print("=" * 60)
print(f"\nFetching data from {START_DATE} to {END_DATE.date()}")
print(f"Stocks: {', '.join(STOCKS)}\n")

all_data = []
for ticker in STOCKS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['Ticker'] = ticker
    df.reset_index(inplace=True)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print(f"\nTotal records fetched: {len(data)}")
print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")

print(f"\nSaving raw data to {RAW_DATA_PATH}...")
data.to_csv(RAW_DATA_PATH, index=False)
print("Done.")
