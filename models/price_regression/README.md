# Price Regression Model

## Objective
Predict the next day's closing stock price using technical indicators.

## Model Type
Random Forest Regressor

## Features
Uses all features from the centralized Feast feature store:
- Price features: Close, High, Low, Open
- Volume features: Volume, Volume_MA_3, Volume_Ratio
- Technical indicators: MA_3, MA_6, MA_8
- Volatility: Volatility_3, Volatility_6
- Percentage changes: Returns, High_Low_Pct, Close_Open_Pct

## Target Variable
`Close` - Next day's closing price

## Training
```bash
python models/price_regression/train.py
```

## Performance Metrics
- RMSE
- MAE
- R² Score
