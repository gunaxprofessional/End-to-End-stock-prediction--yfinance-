import pandas as pd
import numpy as np


def create_features(df):
    """Create technical indicators and features"""
    df = df.copy()

    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

    # Moving averages
    df['MA_3'] = df['Close'].rolling(window=3).mean()
    df['MA_6'] = df['Close'].rolling(window=6).mean()
    df['MA_8'] = df['Close'].rolling(window=8).mean()

    # Volatility
    df['Volatility_3'] = df['Returns'].rolling(window=3).std()
    df['Volatility_6'] = df['Returns'].rolling(window=6).std()

    # Volume features
    df['Volume_MA_3'] = df['Volume'].rolling(window=3).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_3']

    # Target: Next day closing price
    df['Target'] = df['Close'].shift(-1)

    return df


# Load raw data
print("Loading raw data from 'stock_data.csv'...")
data = pd.read_csv(r'data\raw\stock_data.csv')
STOCKS = data['Ticker'].unique()

# Apply feature engineering to each stock using groupby
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

# Save processed data
print("\nSaving processed data to 'stock_data_processed.csv'...")
data.to_csv(r'data\processed\stock_data_processed.csv', index=False)
