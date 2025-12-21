import fastapi
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.pyfunc
import logging

mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = fastapi.FastAPI()
model = mlflow.pyfunc.load_model("models:/StockPricePredictor@champion")

class StockData(BaseModel):
    Date: List[str]
    Open: List[float]
    High: List[float]
    Low: List[float]
    Close: List[float]
    Volume: List[int]

@app.post("/predict")
def predict_stock_prices(data: StockData):
    input_df = pd.DataFrame(data.dict())
    predictions = model.predict(input_df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)