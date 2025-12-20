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
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Volatility
    df['Volatility_5'] = df['Returns'].rolling(window=5).std()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()

    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']

    # Lag features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_2'] = df['Close'].shift(2)
    df['Close_Lag_3'] = df['Close'].shift(3)

    # Target: Next day closing price
    df['Target'] = df['Close'].shift(-1)

    return df


# Load raw data
print("Loading raw data from 'stock_data.csv'...")
data = pd.read_csv(r'data\raw\stock_data.csv')
STOCKS = data['Ticker'].unique()

# Apply feature engineering to each stock
print("Applying feature engineering...")
data_with_features = []
for ticker in STOCKS:
    print(f"Processing features for {ticker}...")
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('Date')
    ticker_data = create_features(ticker_data)
    data_with_features.append(ticker_data)

data = pd.concat(data_with_features)
print(f"Total records after feature engineering: {len(data)}")
print("Ticker counts:", data.groupby('Ticker').size())

# Save processed data
print("\nSaving processed data to 'stock_data_processed.csv'...")
data.dropna(inplace=True)
data.to_csv(r'data\processed\stock_data_processed.csv', index=False)
