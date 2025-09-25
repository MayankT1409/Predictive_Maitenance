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
import pandas as pd

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import preprocessing functions
try:
    from src.preprocess import preprocess_data
    from src.features import add_features
    print("✅ Preprocessing modules imported successfully")
except Exception as e:
    print(f"❌ Error importing preprocessing modules: {e}")
    preprocess_data = None
    add_features = None

app = Flask(__name__)
CORS(app)  # Allow frontend (React) to communicate with backend

# --- Load trained models safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load LightGBM model (primary) and fallback models
models = {}
model_paths = {
    "lgb": "lgb_model.pkl",
    "gb": "gb_model.pkl", 
    "rf": "rf_model.pkl"
}

for model_name, model_file in model_paths.items():
    try:
        model_path = os.path.join(MODELS_DIR, model_file)
        models[model_name] = joblib.load(model_path)
        print(f"✅ {model_name.upper()} model loaded from {model_path}")
    except FileNotFoundError:
        print(f"⚠️  {model_name.upper()} model not found at {model_path}")

# Set primary model (LightGBM preferred)
primary_model = models.get("lgb") or models.get("gb") or models.get("rf")
if primary_model:
    print(f"🎯 Primary model: {'LightGBM' if 'lgb' in models else 'GradientBoosting' if 'gb' in models else 'RandomForest'}")
else:
    print("❌ No models loaded. Prediction API will not work.")

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

# --- Enhanced Prediction API with LightGBM ---
@app.route("/predict", methods=["POST"])
def predict():
    if not primary_model:
        return jsonify({"error": "No models loaded"}), 500
    
    try:
        data = request.json
        
        # Handle both sensor object and feature array formats
        if "sensors" in data:
            # New format: sensor object
            sensors = data["sensors"]
            if isinstance(sensors, dict):
                # Convert sensor dict to feature array using preprocessing
                if preprocess_data is None or add_features is None:
                    return jsonify({"error": "Preprocessing modules not available"}), 500
                
                # Create DataFrame from sensor data
                df = pd.DataFrame([sensors])
                
                # Preprocess and add features
                df_processed = preprocess_data(df)
                df_features = add_features(df_processed, include_trend=False)
                
                # Load feature columns to ensure correct order
                try:
                    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
                    df_features = df_features.reindex(columns=feature_columns, fill_value=0)
                except:
                    pass
                
                # Convert to numpy array
                features_array = df_features.values
            else:
                # Legacy format: feature array
                features_array = np.array(sensors).reshape(1, -1)
        else:
            return jsonify({"error": "No sensor data provided"}), 400
        
        # Get predictions from all available models
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(features_array)[0]
                prob = model.predict_proba(features_array)[0, 1] if hasattr(model, 'predict_proba') else None
                predictions[model_name] = int(pred)
                probabilities[model_name] = float(prob) if prob is not None else None
            except Exception as e:
                print(f"Error with {model_name} model: {e}")
                continue
        
        # Use primary model for main prediction
        primary_name = "lgb" if "lgb" in models else "gb" if "gb" in models else "rf"
        main_prediction = predictions.get(primary_name, 0)
        main_probability = probabilities.get(primary_name, 0.5)
        
        # Calculate risk metrics
        risk_score = main_probability
        risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
        health_score = int((1 - risk_score) * 100)
        
        # Estimate remaining days (simplified heuristic)
        remaining_days = max(1, int((1 - risk_score) * 30)) if main_prediction == 0 else 0
        
        # Generate recommendations based on risk level
        if risk_level == "High":
            recommendations = [
                "Schedule immediate inspection",
                "Stop equipment operation if possible",
                "Check for abnormal vibrations or sounds",
                "Review recent maintenance logs"
            ]
        elif risk_level == "Medium":
            recommendations = [
                "Schedule maintenance within 1-2 weeks",
                "Increase monitoring frequency",
                "Check lubrication levels",
                "Review sensor trends"
            ]
        else:
            recommendations = [
                "Continue routine monitoring",
                "Adhere to scheduled maintenance",
                "Monitor for trend changes",
                "Keep maintenance logs updated"
            ]
        
        return jsonify({
            "prediction": main_prediction,
            "probability": main_probability,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "health_score": health_score,
            "remaining_days": remaining_days,
            "recommendations": recommendations,
            "model_used": primary_name.upper(),
            "all_predictions": predictions,
            "all_probabilities": probabilities
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

# --- File Upload Prediction API ---
@app.route("/predict_file", methods=["POST"])
def predict_file():
    if not primary_model:
        return jsonify({"error": "No models loaded"}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are supported"}), 400
        
        # Read CSV file
        import pandas as pd
        from io import StringIO
        
        # Read file content
        file_content = file.read().decode('utf-8')
        df = pd.read_csv(StringIO(file_content))
        
        # Preprocess data
        if preprocess_data is None or add_features is None:
            return jsonify({"error": "Preprocessing modules not available"}), 500
        
        df_processed = preprocess_data(df)
        df_features = add_features(df_processed, include_trend=False)
        
        # Load feature columns to ensure correct order
        try:
            feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
            df_features = df_features.reindex(columns=feature_columns, fill_value=0)
        except:
            pass
        
        # Get predictions for all rows
        predictions = []
        probabilities = []
        
        for _, row in df_features.iterrows():
            row_array = row.values.reshape(1, -1)
            
            # Get prediction from primary model
            pred = primary_model.predict(row_array)[0]
            prob = primary_model.predict_proba(row_array)[0, 1] if hasattr(primary_model, 'predict_proba') else 0.5
            
            predictions.append(int(pred))
            probabilities.append(float(prob))
        
        # Calculate aggregate metrics
        avg_probability = sum(probabilities) / len(probabilities)
        failure_count = sum(predictions)
        total_rows = len(predictions)
        
        return jsonify({
            "predictions": predictions,
            "probabilities": probabilities,
            "summary": {
                "total_rows": total_rows,
                "failure_predictions": failure_count,
                "average_failure_probability": avg_probability,
                "risk_level": "High" if avg_probability > 0.7 else "Medium" if avg_probability > 0.3 else "Low"
            }
        })
    
    except Exception as e:
        print(f"File prediction error: {e}")
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
