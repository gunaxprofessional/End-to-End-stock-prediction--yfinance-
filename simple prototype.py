import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Set dates to month boundaries
END_DATE = datetime.now().replace(
    day=1) - timedelta(days=1)  # Last day of previous month
# First day of month 12 months ago
START_DATE = (END_DATE - timedelta(days=365)).replace(day=1)

print("=" * 60)
print("STOCK PRICE PREDICTION - NEXT DAY CLOSING")
print("=" * 60)
print(f"\nFetching data from {START_DATE.date()} to {END_DATE.date()}")
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

# Feature Engineering


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


# Apply feature engineering to each stock
data_with_features = []
for ticker in STOCKS:
    ticker_data = data[data['Ticker'] == ticker].copy()
    ticker_data = create_features(ticker_data)
    data_with_features.append(ticker_data)

data = pd.concat(data_with_features)
data.dropna(inplace=True)

# Define feature columns
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns',
                'High_Low_Pct', 'Close_Open_Pct', 'MA_5', 'MA_10', 'MA_20',
                'Volatility_5', 'Volatility_10', 'Volume_MA_5', 'Volume_Ratio',
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']

# Time-based split using month boundaries
data = data.sort_values('Date').reset_index(drop=True)

# Get unique months
data['YearMonth'] = data['Date'].dt.to_period('M')
unique_months = sorted(data['YearMonth'].unique())
total_months = len(unique_months)

print(f"\nTotal months available: {total_months}")

# Split by months: 9 months train, 2 months test, 1 month validation
train_months = unique_months[:9]
test_months = unique_months[9:11]
val_months = unique_months[11:]

# Split data based on months
train_data = data[data['YearMonth'].isin(train_months)]
test_data = data[data['YearMonth'].isin(test_months)]
val_data = data[data['YearMonth'].isin(val_months)]

print("\n" + "=" * 60)
print("TIME-BASED DATA SPLIT (BY MONTH)")
print("=" * 60)
print(
    f"Training Set:   {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
print(
    f"                Months: {', '.join(str(m) for m in train_months)} ({len(train_months)} months, {len(train_data)} samples)")
print(
    f"\nTest Set:       {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")
print(
    f"                Months: {', '.join(str(m) for m in test_months)} ({len(test_months)} months, {len(test_data)} samples)")
print(
    f"\nValidation Set: {val_data['Date'].min().date()} to {val_data['Date'].max().date()}")
print(
    f"                Months: {', '.join(str(m) for m in val_months)} ({len(val_months)} months, {len(val_data)} samples)")

# Prepare data
X_train = train_data[feature_cols]
y_train = train_data['Target']

X_test = test_data[feature_cols]
y_test = test_data['Target']

X_val = val_data[feature_cols]
y_val = val_data['Target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Train model
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST REGRESSOR")
print("=" * 60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("✓ Model training completed")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_val_pred = model.predict(X_val_scaled)

# Evaluation function


def evaluate(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{dataset_name}:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")


# Results
print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

evaluate(y_train, y_train_pred, "Training Set")
evaluate(y_test, y_test_pred, "Test Set")
evaluate(y_val, y_val_pred, "Validation Set (Out-of-Sample)")

# Feature importance
print("\n" + "=" * 60)
print("TOP 10 FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['Feature']:20s}: {row['Importance']:.4f}")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (VALIDATION SET)")
print("=" * 60)

val_sample = val_data.head(10).copy()
val_sample['Predicted_Close'] = y_val_pred[:10]
val_sample['Actual_Close'] = val_sample['Target']
val_sample['Error'] = val_sample['Predicted_Close'] - \
    val_sample['Actual_Close']
val_sample['Error_Pct'] = (val_sample['Error'] /
                           val_sample['Actual_Close']) * 100

print(val_sample[['Date', 'Ticker', 'Actual_Close',
      'Predicted_Close', 'Error', 'Error_Pct']].to_string(index=False))

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
