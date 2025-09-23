# src/features.py
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame, include_trend: bool = False) -> pd.DataFrame:
    """
    Add extra features for predictive maintenance with stronger domain features.
    """
    df = df.copy()

    # Enhanced rolling statistics with multiple windows
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "failure"]
    feature_frames = [df]
    
    for col in numeric_cols:
        # Multiple rolling windows for better temporal patterns
        for window in [3, 7, 12, 24]:
            r = df[col].rolling(window=window, min_periods=1)
            new_cols = {
                f"{col}_mean_{window}": r.mean(),
                f"{col}_std_{window}": r.std().fillna(0),
                f"{col}_max_{window}": r.max(),
                f"{col}_min_{window}": r.min(),
                f"{col}_median_{window}": r.median(),
            }
            feature_frames.append(pd.DataFrame(new_cols))
        
        # Lag features for temporal dependencies
        lag_dict = {}
        for lag in [1, 2, 3, 6, 12]:
            lag_col = f"{col}_lag_{lag}"
            lag_dict[lag_col] = df[col].shift(lag)
            lag_dict[f"{col}_delta_{lag}"] = df[col] - lag_dict[lag_col]
        feature_frames.append(pd.DataFrame(lag_dict))
        
        # Change rate features
        change_dict = {
            f"{col}_pct_change": df[col].pct_change().fillna(0),
            f"{col}_diff": df[col].diff().fillna(0),
        }
        feature_frames.append(pd.DataFrame(change_dict))
        
        # Trend features (optional; can be very slow on large datasets)
        if include_trend:
            trend_dict = {
                f"{col}_trend_5": (
                    df[col]
                    .rolling(5, min_periods=2)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
                    .fillna(0)
                ),
                f"{col}_trend_10": (
                    df[col]
                    .rolling(10, min_periods=2)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
                    .fillna(0)
                ),
            }
            feature_frames.append(pd.DataFrame(trend_dict))

    # Advanced interaction features
    if {"temperature", "vibration"}.issubset(df.columns):
        feature_frames.append(pd.DataFrame({
            "temp_vib_ratio": df["temperature"] / (df["vibration"] + 1e-6),
            "temp_vib_product": df["temperature"] * df["vibration"],
            "temp_vib_diff": df["temperature"] - df["vibration"],
        }))
    
    if {"pressure", "torque"}.issubset(df.columns):
        feature_frames.append(pd.DataFrame({
            "press_torque_ratio": df["pressure"] / (df["torque"] + 1e-6),
            "press_torque_product": df["pressure"] * df["torque"],
            "press_torque_diff": df["pressure"] - df["torque"],
        }))
    
    if {"temperature", "pressure"}.issubset(df.columns):
        feature_frames.append(pd.DataFrame({
            "temp_press_ratio": df["temperature"] / (df["pressure"] + 1e-6),
            "temp_press_product": df["temperature"] * df["pressure"],
        }))
    
    if {"vibration", "pressure"}.issubset(df.columns):
        feature_frames.append(pd.DataFrame({
            "vib_press_ratio": df["vibration"] / (df["pressure"] + 1e-6),
            "vib_press_product": df["vibration"] * df["pressure"],
        }))

    # Statistical features across all sensors
    sensor_cols = [c for c in numeric_cols if c in ["temperature", "vibration", "pressure", "rpm", "torque", "humidity"]]
    if len(sensor_cols) > 1:
        sensor_stats = pd.DataFrame({
            "sensor_mean": df[sensor_cols].mean(axis=1),
            "sensor_std": df[sensor_cols].std(axis=1).fillna(0),
            "sensor_max": df[sensor_cols].max(axis=1),
            "sensor_min": df[sensor_cols].min(axis=1),
        })
        sensor_stats["sensor_range"] = sensor_stats["sensor_max"] - sensor_stats["sensor_min"]
        feature_frames.append(sensor_stats)
        
        # Cross-sensor correlations
        for i, col1 in enumerate(sensor_cols):
            for col2 in sensor_cols[i+1:]:
                feature_frames.append(pd.DataFrame({
                    f"{col1}_{col2}_corr": df[col1].rolling(10).corr(df[col2]).fillna(0)
                }))

    # Time-based features if timestamp exists
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        time_feats = pd.DataFrame({
            "hour": ts.dt.hour,
            "day_of_week": ts.dt.dayofweek,
        })
        time_feats["is_weekend"] = (time_feats["day_of_week"] >= 5).astype(int)
        time_feats["hour_sin"] = np.sin(2 * np.pi * time_feats["hour"] / 24)
        time_feats["hour_cos"] = np.cos(2 * np.pi * time_feats["hour"] / 24)
        feature_frames.append(time_feats)

    # Concatenate all feature blocks at once to avoid fragmentation warnings
    df = pd.concat(feature_frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    # Replace infinities from pct_change/divisions and fill remaining NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


