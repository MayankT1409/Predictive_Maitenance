# src/predict.py
import joblib
import pandas as pd
from src.preprocess import preprocess_data
from src.features import add_features

# Load model once (when FastAPI starts)
MODEL_PATH = "models/gb_model.pkl"
model = joblib.load(MODEL_PATH)

# Load the feature columns used during training
FEATURES_PATH = "models/feature_columns.pkl"
feature_columns = joblib.load(FEATURES_PATH)

def prepare_data(data: pd.DataFrame, drop_target=True) -> pd.DataFrame:
    """
    Preprocess + feature engineering for prediction
    """
    data = preprocess_data(data)
    data = add_features(data)

    if drop_target and "failure" in data.columns:
        data = data.drop("failure", axis=1)

    # Keep only trained features
    data = data.reindex(columns=feature_columns, fill_value=0)
    return data

def predict_new(data: pd.DataFrame):
    """
    Returns predictions and probabilities
    """
    X = prepare_data(data)
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return predictions, probs
