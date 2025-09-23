# # app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import joblib
# import numpy as np
# from src.preprocess import preprocess_data
# from src.features import add_features

# # -----------------------------
# # Load model, scaler, features
# # -----------------------------
# MODEL_PATH = "models/predictive_model.pkl"
# SCALER_PATH = "models/scaler.pkl"
# FEATURES_PATH = "models/feature_columns.pkl"

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)
# feature_columns = joblib.load(FEATURES_PATH)

# # -----------------------------
# # FastAPI setup
# # -----------------------------
# app = FastAPI(title="Predictive Maintenance API")

# # Pydantic model for input validation
# class SensorData(BaseModel):
#     temperature: float
#     vibration: float
#     pressure: float
#     rpm: float
#     temp_change: float = 0.0
#     vib_change: float = 0.0
#     Timestamp: str = None
#     torque: float = 0.0  # optional, for feature engineering

# # -----------------------------
# # Helper function
# # -----------------------------
# def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess, feature engineer, scale, and align columns."""
#     df = preprocess_data(data)
#     df = add_features(df)

#     # Keep only trained features and fill missing with 0
#     df = df.reindex(columns=feature_columns, fill_value=0)

#     # Scale
#     df_scaled = scaler.transform(df)
#     return df_scaled

# # -----------------------------
# # API endpoint
# # -----------------------------
# @app.post("/predict")
# def predict(data: SensorData):
#     df = pd.DataFrame([data.dict()])
#     X_prepared = prepare_input(df)

#     prediction = model.predict(X_prepared)[0]
#     probability = model.predict_proba(X_prepared)[:, 1][0]

#     return {
#         "prediction": int(prediction),
#         "probability": float(probability)
#     }

# # -----------------------------
# # Run server
# # -----------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
from pydantic import BaseModel

class InputData(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    humidity: float
    rpm: float
    torque: float
    timestamp: str
    # add raw feaures that your preprocess_data() + add_features() expect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build absolute path to models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load all models
models = {
    "rf": joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
    "gb": joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl")),
    "xgb": joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl")),
    "lgb": joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl")),
}

@app.post("/predict/{model_name}")
def predict(model_name: str, data: InputData):
    if model_name not in models:
        return {"error": f"Model '{model_name}' not found. Available: {list(models.keys())}"}
    
    df = pd.DataFrame([data.dict()])
    prediction = models[model_name].predict(df)
    return {"model": model_name, "prediction": prediction.tolist()}
