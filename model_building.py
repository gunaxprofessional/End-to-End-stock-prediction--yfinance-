import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import datetime

# --- CONFIGURATION ---
MODEL_NAME = "StockPricePredictor"
EXPERIMENT_NAME = "Stock_Price_Prediction"

# Set tracking URI to use SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_data():
    """Load processed data from CSV file."""
    print("Loading data...")
    data_path = os.path.join('data', 'processed', 'stock_data_processed.csv')
    try:
        data = pd.read_csv(data_path)
        print(f"Data loaded successfully. Total records: {len(data)}")
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def split_data(data):
    """Split data into training and testing sets based on date ranges."""
    train_start_date = pd.to_datetime('2024-11-01')
    train_end_date = pd.to_datetime('2025-08-31')
    test_start_date = pd.to_datetime('2025-09-01')
    test_end_date = pd.to_datetime('2025-10-31')
    
    data['Date'] = pd.to_datetime(data['Date'])

    train_data = data[(data['Date'] >= train_start_date) & (data['Date'] <= train_end_date)].copy()
    test_data = data[(data['Date'] >= test_start_date) & (data['Date'] <= test_end_date)].copy()

    if train_data.empty or test_data.empty:
        raise ValueError("Train or test data is empty. Please check the date ranges and data availability.")

    print(f"Training data records: {len(train_data)}")
    print(f"Testing data records: {len(test_data)}")
    
    return train_data, test_data

    
def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return rmse, mae, r2

def model_training(train_data, test_data):
    """Train model and log metadata/governance to MLflow."""
    print("Starting model training...")
    client = MlflowClient()
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_cols = [col for col in train_data.columns if col not in ['Date', 'Ticker', 'Target']]
    print(f"Feature columns: {feature_cols}")
    X_train, y_train = train_data[feature_cols], train_data['Target']
    X_test, y_test = test_data[feature_cols], test_data['Target']

    # 1. Define Tags (Team/Governance)
    tags = {
        "team": "Alpha-Quant",
        "project": "Market-Forecasting",
        "priority": "High",
        "created_by": os.getenv("USER", "CI_CD_Pipeline"),
        "framework": "scikit-learn"
    }

    with mlflow.start_run(run_name=f"RF_Train_{datetime.date.today()}") as run:
        print("MLflow run started...")
        mlflow.set_tags(tags)
        
        # Model Training
        print("Training RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model training completed.")

        

        y_pred = model.predict(X_test)
        print("Predictions on test set completed.")

        # Metrics
        rmse, mae, r2 = calculate_metrics(y_test, y_pred)

        # 2. Infer Model Signature (Schema Enforcement)
        signature = infer_signature(X_test.astype(float), y_pred)

        # 3. Log Model with Registration
        # This automatically creates the model version and returns details
        print("Logging model to MLflow...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="stock_model_artifact",
            signature=signature,
            input_example=X_test.iloc[:3].astype(float),
            registered_model_name=MODEL_NAME
        )
        print("Model logged successfully.")

        # 4. Log Metrics & Params
        # Move description updates before logging metrics
        version = model_info.registered_model_version
        
        # Set Registered Model Description (General)
        print("Updating registered model description...")
        client.update_registered_model(
            name=MODEL_NAME,
            description="Production model for predicting daily stock price movements."
        )

        # Set Version-Specific Description (Audit Trail)
        print("Updating model version description...")
        client.update_model_version(
            name=MODEL_NAME,
            version=version,
            description=f"Automated Training. R2: {r2:.4f}. Dataset size: {len(train_data)} rows."
        )

        print("Logging metrics and parameters...")
        mlflow.log_params({"n_estimators": 100, "features_count": len(feature_cols)})
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})

        # 6. Promotion Logic (Alias-based) - Challenge the Champion
        print("--- Challenge the Champion ---")
        champion_r2 = -1.0
        try:
            # Fetch current champion details
            champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champion_run_id = champion_version.run_id
            champion_run = client.get_run(champion_run_id)
            champion_r2 = champion_run.data.metrics.get("R2", 0.0)
            print(f"Current Champion Version: {champion_version.version}, R2: {champion_r2:.4f}")
        except Exception:
            # If no champion exists (first run), set baseline
            print("No existing champion found. Initializing baseline.")

        # Promotion check
        if r2 > 0.85 and r2 > champion_r2:
            print(f"New Model R2 ({r2:.4f}) > Champion R2 ({champion_r2:.4f}). Promoting...")
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
            print(f"Version {version} is now the CHAMPION.")
        else:
            print(f"New Model R2 ({r2:.4f}) <= Champion R2 ({champion_r2:.4f}) or Threshold (0.85).")
            print(f"Version {version} remains a candidate.")

        print("Model training and logging completed.")

        try:
            print("Logging SHAP background dataset...")
            # logging sample data for shap
            background = X_train.sample(n=min(100, len(X_train)), random_state=42)

            # Save background dataset
            background_path = "shap_background.parquet"
            background.to_parquet(background_path)

            # Log as MLflow artifact
            mlflow.log_artifact(background_path, artifact_path="shap")
        except Exception as e:
            print(f"Failed to log SHAP background dataset: {e}")

        return run.info.run_id

if __name__ == "__main__":
    try:
        print("Pipeline started...")
        raw_data = load_data()
        train, test = split_data(raw_data)
        run_id = model_training(train, test)
        print(f"Pipeline finished successfully. Run ID: {run_id}")
    except Exception as e:
        print(f"Pipeline failed: {e}")