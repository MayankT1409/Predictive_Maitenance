from fastapi import FastAPI
import uvicorn
import pandas as pd
from src.predict import predict_new  # ✅ now using updated function

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])  # Convert JSON → DataFrame
    predictions, probs = predict_new(df)
    return {
        "prediction": int(predictions[0]),
        "probability": float(probs[0])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
