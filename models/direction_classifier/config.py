# Direction Classifier Model Configuration

MODEL_NAME = "StockDirectionClassifier"
EXPERIMENT_NAME = "Stock_Direction_Classification"

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42,
    'class_weight': 'balanced'  # Handle class imbalance
}

# Training configuration
TRAIN_START_DATE = '2024-11-01'
TRAIN_END_DATE = '2025-08-31'
TEST_START_DATE = '2025-09-01'
TEST_END_DATE = '2025-10-31'

MODEL_ALIAS = "champion"

# Classification threshold
# Target: 1 if next day's close > today's close (UP), else 0 (DOWN)
TARGET = "Direction"
