# src/features.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra features for predictive maintenance.
    You can expand this as needed.
    """
    df = df.copy()

    # Example rolling statistics (good for sensor data)
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ["failure"]:  # avoid target column
            df[f"{col}_mean_3"] = df[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_std_3"] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
            df[f"{col}_max_3"] = df[col].rolling(window=3, min_periods=1).max()

            df[f"{col}_mean_7"] = df[col].rolling(window=7, min_periods=1).mean()
            df[f"{col}_std_7"] = df[col].rolling(window=7, min_periods=1).std().fillna(0)
            df[f"{col}_max_7"] = df[col].rolling(window=7, min_periods=1).max()

    # Example interaction features (domain-specific, can improve accuracy)
    if {"temperature", "vibration"}.issubset(df.columns):
        df["temp_vibration_ratio"] = df["temperature"] / (df["vibration"] + 1e-6)

    if {"pressure", "torque"}.issubset(df.columns):
        df["pressure_torque_product"] = df["pressure"] * df["torque"]

    return df
