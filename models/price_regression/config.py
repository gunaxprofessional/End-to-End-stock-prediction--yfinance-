# Price Regression Model Configuration

MODEL_NAME = "StockPricePredictor"
EXPERIMENT_NAME = "Stock_Price_Prediction"

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Training configuration
TRAIN_START_DATE = '2024-11-01'
TRAIN_END_DATE = '2025-08-31'
TEST_START_DATE = '2025-09-01'
TEST_END_DATE = '2025-10-31'

MODEL_ALIAS = "champion"

# Target variable
TARGET = "Close"
