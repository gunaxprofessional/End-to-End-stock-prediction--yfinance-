# Direction Classifier Model

## Objective
Predict whether the stock price will go UP or DOWN the next day.

## Model Type
Random Forest Classifier

## Features
Uses the same features from the centralized Feast feature store as the price regression model:
- Price features: Close, High, Low, Open
- Volume features: Volume, Volume_MA_3, Volume_Ratio
- Technical indicators: MA_3, MA_6, MA_8
- Volatility: Volatility_3, Volatility_6
- Percentage changes: Returns, High_Low_Pct, Close_Open_Pct

## Target Variable
`Direction` - Binary classification:
- **1 (UP)**: Next day's close > current day's close
- **0 (DOWN)**: Next day's close <= current day's close

## Training
```bash
python models/direction_classifier/train.py
```

## Performance Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Use Cases
- Trading signals (buy/sell recommendations)
- Risk assessment
- Portfolio optimization
- Market sentiment analysis
