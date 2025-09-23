# # # app.py
# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # import pandas as pd
# # import joblib
# # import numpy as np
# # from src.preprocess import preprocess_data
# # from src.features import add_features

# # # -----------------------------
# # # Load model, scaler, features
# # # -----------------------------
# # MODEL_PATH = "models/predictive_model.pkl"
# # SCALER_PATH = "models/scaler.pkl"
# # FEATURES_PATH = "models/feature_columns.pkl"

# # model = joblib.load(MODEL_PATH)
# # scaler = joblib.load(SCALER_PATH)
# # feature_columns = joblib.load(FEATURES_PATH)

# # # -----------------------------
# # # FastAPI setup
# # # -----------------------------
# # app = FastAPI(title="Predictive Maintenance API")

# # # Pydantic model for input validation
# # class SensorData(BaseModel):
# #     temperature: float
# #     vibration: float
# #     pressure: float
# #     rpm: float
# #     temp_change: float = 0.0
# #     vib_change: float = 0.0
# #     Timestamp: str = None
# #     torque: float = 0.0  # optional, for feature engineering

# # # -----------------------------
# # # Helper function
# # # -----------------------------
# # def prepare_input(data: pd.DataFrame) -> pd.DataFrame:
# #     """Preprocess, feature engineer, scale, and align columns."""
# #     df = preprocess_data(data)
# #     df = add_features(df)

# #     # Keep only trained features and fill missing with 0
# #     df = df.reindex(columns=feature_columns, fill_value=0)

# #     # Scale
# #     df_scaled = scaler.transform(df)
# #     return df_scaled

# # # -----------------------------
# # # API endpoint
# # # -----------------------------
# # @app.post("/predict")
# # def predict(data: SensorData):
# #     df = pd.DataFrame([data.dict()])
# #     X_prepared = prepare_input(df)

# #     prediction = model.predict(X_prepared)[0]
# #     probability = model.predict_proba(X_prepared)[:, 1][0]

# #     return {
# #         "prediction": int(prediction),
# #         "probability": float(probability)
# #     }

# # # -----------------------------
# # # Run server
# # # -----------------------------
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import joblib
# import os
# import pandas as pd
# from pydantic import BaseModel

# class InputData(BaseModel):
#     temperature: float
#     pressure: float
#     vibration: float
#     humidity: float
#     rpm: float
#     torque: float
#     timestamp: str
#     # add raw feaures that your preprocess_data() + add_features() expect

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Build absolute path to models directory
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# # Load all models
# models = {
#     "rf": joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
#     "gb": joblib.load(os.path.join(MODELS_DIR, "gb_model.pkl")),
#     "xgb": joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl")),
# }

# @app.post("/predict/{model_name}")
# def predict(model_name: str, data: InputData):
#     if model_name not in models:
#         return {"error": f"Model '{model_name}' not found. Available: {list(models.keys())}"}
    
#     df = pd.DataFrame([data.dict()])
#     prediction = models[model_name].predict(df)
#     return {"model": model_name, "prediction": prediction.tolist()}


from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import joblib

app = Flask(__name__)
CORS(app)  # Allow frontend (React) to communicate with backend

# --- Load trained model safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gb_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    print(f"Model not found at {MODEL_PATH}. Prediction API will not work until model is loaded.")

# --- Home route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Predictive Maintenance API is running!",
        "endpoints": {
            "predict": "/predict (POST with JSON: {\"sensors\": [...]})",
            "sensor_data": "/api/sensors (POST with JSON)"
        }
    })

# --- Prediction API ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        features = data.get("sensors", [])
        
        # --- Check feature length ---
        expected_features = model.n_features_in_ if hasattr(model, "n_features_in_") else 48
        if len(features) != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, got {len(features)}"}), 400
        
        # --- Predict ---
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Sensor Data API ---
@app.route("/api/sensors", methods=["POST"])
def receive_sensors():
    try:
        data = request.json
        # Optionally save sensor data to DB, CSV, or logs
        return jsonify({"status": "success", "data_received": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Run Server ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
