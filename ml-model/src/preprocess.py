# import pandas as pd
# import numpy as np

# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     # 1) Rename columns to canonical names your code uses
#     df = df.rename(columns={
#         "Temperature (°C)": "temperature",
#         "Vibration (mm/s)": "vibration",
#         "Pressure (Pa)": "pressure",
#         "RPM": "rpm",
#         "Maintenance Required": "failure",
#         "Temp_Change": "temp_change",
#         "Vib_Change": "vib_change",
#     })

#     # 2) Timestamp -> numeric time features (then drop the raw column)
#     if "Timestamp" in df.columns:
#         ts = pd.to_datetime(df["Timestamp"], errors="coerce")
#         # cyclic encodings help tree models when patterns repeat daily/weekly
#         df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
#         df["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
#         df["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
#         df = df.drop(columns=["Timestamp"])

#     # 3) Ensure numeric types
#     for c in ["temperature", "vibration", "pressure", "rpm", "temp_change", "vib_change"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#     if "failure" in df.columns:
#         df["failure"] = pd.to_numeric(df["failure"], errors="coerce").astype(int)

#     # 4) Handle missing values (NO backward fill)
#     df = df.ffill()

#     # 5) Lag/delta/rolling features (trailing windows only)
#     base = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df.columns]
#     for c in base:
#         for lag in (1, 3, 6):
#             df[f"{c}_lag{lag}"]   = df[c].shift(lag)
#             df[f"{c}_delta{lag}"] = df[c] - df[f"{c}_lag{lag}"]

#         for w in (3, 12, 48):  # tweak to your sampling rate
#             r = df[c].rolling(window=w, min_periods=w)
#             df[f"{c}_mean{w}"] = r.mean()
#             df[f"{c}_std{w}"]  = r.std()
#             df[f"{c}_min{w}"]  = r.min()
#             df[f"{c}_max{w}"]  = r.max()

#     # 6) Drop initial rows with NaNs from lags/rollings
#     df = df.dropna().reset_index(drop=True)
#     return df


# import pandas as pd
# import numpy as np

# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     # 1) Rename columns (standard names)
#     df = df.rename(columns={
#         "Temperature (°C)": "temperature",
#         "Vibration (mm/s)": "vibration",
#         "Pressure (Pa)": "pressure",
#         "RPM": "rpm",
#         "Maintenance Required": "failure",
#         "Temp_Change": "temp_change",
#         "Vib_Change": "vib_change",
#     })

#     # 2) Timestamp -> cyclic time features (hour/day of week)
#     if "Timestamp" in df.columns:
#         ts = pd.to_datetime(df["Timestamp"], errors="coerce")
#         df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
#         df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
#         df["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
#         df["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
#         df = df.drop(columns=["Timestamp"])

#     # 3) Convert to numeric safely
#     numeric_cols = ["temperature", "vibration", "pressure", "rpm", "temp_change", "vib_change"]
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     if "failure" in df.columns:
#         df["failure"] = pd.to_numeric(df["failure"], errors="coerce").astype(int)

#     # 4) Handle missing values more carefully
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = df[c].interpolate(method="linear").fillna(df[c].median())

#     # 5) Add **lightweight lag/rolling features** (reduced)
#     base = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df.columns]
#     for c in base:
#         # Short-term lag (1 step)
#         df[f"{c}_lag1"] = df[c].shift(1)
#         df[f"{c}_delta1"] = df[c] - df[f"{c}_lag1"]

#         # Short rolling mean/std
#         df[f"{c}_mean3"] = df[c].rolling(window=3, min_periods=1).mean()
#         df[f"{c}_std3"]  = df[c].rolling(window=3, min_periods=1).std()

#     # 6) Drop rows with NaN after lagging
#     df = df.dropna().reset_index(drop=True)

#     return df



import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Rename columns (standard names)
    df = df.rename(columns={
        "Temperature (°C)": "temperature",
        "Vibration (mm/s)": "vibration",
        "Pressure (Pa)": "pressure",
        "RPM": "rpm",
        "Maintenance Required": "failure",
        "Temp_Change": "temp_change",
        "Vib_Change": "vib_change",
    })

    # 2) Timestamp -> cyclic time features (hour/day of week)
    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
        df = df.drop(columns=["Timestamp"])

    # 3) Convert to numeric safely
    numeric_cols = ["temperature", "vibration", "pressure", "rpm", "temp_change", "vib_change"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "failure" in df.columns:
        df["failure"] = pd.to_numeric(df["failure"], errors="coerce").astype(int)

    # 4) Handle missing values carefully
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].interpolate(method="linear").fillna(df[c].median())

    # 5) Add lag/rolling features
    base = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in df.columns]
    for c in base:
        # Short-term lag (1 step)
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_delta1"] = df[c] - df[f"{c}_lag1"]

        # Short rolling stats (3-step)
        df[f"{c}_mean3"] = df[c].rolling(window=3, min_periods=1).mean()
        df[f"{c}_std3"]  = df[c].rolling(window=3, min_periods=1).std()

        # Longer rolling stats (10-step)
        df[f"{c}_mean10"] = df[c].rolling(window=10, min_periods=1).mean()
        df[f"{c}_std10"]  = df[c].rolling(window=10, min_periods=1).std()

    # 6) Interaction features (captures combined stress effects)
    if set(["temperature", "vibration", "pressure"]).issubset(df.columns):
        df["temp_vib"] = df["temperature"] * df["vibration"]
        df["temp_press"] = df["temperature"] * df["pressure"]
        df["vib_press"] = df["vibration"] * df["pressure"]

    # 7) Failure history (optional but useful)
    if "failure" in df.columns:
        df["failure_lag1"] = df["failure"].shift(1).fillna(0).astype(int)
        df["failure_cum"]  = df["failure"].cumsum()

    # 8) Drop rows with NaN after lagging
    df = df.dropna().reset_index(drop=True)

    return df
