import pandas as pd

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.global_config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FEATURE_PARQUET_PATH


def create_features(df):
    df = df.copy()

    df['Returns'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_6'] = df['Close'].rolling(window=6).mean()
    df['MA_8'] = df['Close'].rolling(window=8).mean()

    df['Volatility_3'] = df['Returns'].rolling(window=3).std()
    df['Volatility_6'] = df['Returns'].rolling(window=6).std()

    df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_3']

    df['Target'] = df['Close'].shift(-1)

    return df


print(f"Loading raw data from {RAW_DATA_PATH}...")
data = pd.read_csv(RAW_DATA_PATH)

# Ensure numeric columns are properly typed
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

STOCKS = data['Ticker'].unique()

print("Applying feature engineering...")
data_with_features = []

grouped_data = data.groupby('Ticker')
for ticker, ticker_data in grouped_data:
    print(f"Processing features for {ticker}...")
    ticker_data = ticker_data.sort_values('Date')
    ticker_data = create_features(ticker_data)
    data_with_features.append(ticker_data)

data = pd.concat(data_with_features)
print(f"Total records after feature engineering: {len(data)}")
print("Ticker counts:", data.groupby('Ticker').size())

data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize('UTC')
data['ticker'] = data['Ticker']
data['created_timestamp'] = pd.Timestamp.now(tz='UTC')

print(f"\nSaving processed data to {PROCESSED_DATA_PATH}...")
data.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Saving parquet to {FEATURE_PARQUET_PATH}...")
data.to_parquet(FEATURE_PARQUET_PATH, index=False)
print("Done.")
