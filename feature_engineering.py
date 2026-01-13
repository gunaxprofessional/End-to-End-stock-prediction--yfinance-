import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO



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


from storage import MinioArtifactStore

from config import RAW_DATA_KEY, PROCESSED_DATA_KEY, FEATURE_PARQUET_KEY

print("Loading raw data from MinIO...")
store = MinioArtifactStore()
data = store.load_df(RAW_DATA_KEY)

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

print("\nSaving processed data to MinIO...")
store.save_df(data, PROCESSED_DATA_KEY)

parquet_buffer = BytesIO()
data.to_parquet(parquet_buffer, index=False)
store.s3_client.put_object(
    Bucket=store.bucket_name,
    Key=FEATURE_PARQUET_KEY,
    Body=parquet_buffer.getvalue()
)
print("Saved stock_features.parquet to MinIO")
