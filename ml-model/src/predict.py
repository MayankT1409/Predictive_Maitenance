import joblib
import pandas as pd
from .preprocess import preprocess_data
from .train import add_features   # ✅ import feature engineering

def predict_new(data: pd.DataFrame, model_path="models/gb_model.pkl"):
    # Load model
    model = joblib.load(model_path)

    # Preprocess
    data = preprocess_data(data)
    data = add_features(data)

    # Drop target if present
    if "failure" in data.columns:
        data = data.drop("failure", axis=1)

    # Predict
    predictions = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    return predictions, probs
