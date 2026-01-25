import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Try to automatically detect the target column.
    """
    candidates = [
        "failure",
        "machine_failure",
        "Machine failure",
        "Target",
        "target",
        "maintenance_required",
        "Maintenance Required",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # AI4I-style: if failure types exist, create binary target
    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    if all(ft in df.columns for ft in failure_types):
        df["binary_failure"] = (df[failure_types].sum(axis=1) > 0).astype(int)
        return "binary_failure"

    raise ValueError(
        f"No known target column found.\nAvailable columns:\n{list(df.columns)}"
    )


def main(csv_path: str):
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(csv_path)

    # ---------------- TARGET ----------------
    target_col = detect_target_column(df)
    print("Detected target column:", target_col)

    y = df[target_col]

    # ---------------- FEATURES ----------------
    drop_cols = [
        target_col,
        "UDI",
        "Product ID",
        "product_id",
        "machine_id",
        "id",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    print("Number of features used:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts())

    # ---------------- SCALING ----------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- TRAIN / TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------- HANDLE IMBALANCE ----------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE class distribution:")
    print(pd.Series(y_train_res).value_counts())

    # ---------------- MODEL ----------------
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    gb.fit(X_train_res, y_train_res)

    # ---------------- EVALUATION ----------------
    y_pred = gb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc * 100, 2), "%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV file")
    args = parser.parse_args()

    main(args.csv)
