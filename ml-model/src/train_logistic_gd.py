import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def detect_target_column(df):
    for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found")


def main(csv_path):
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]

    # ---------------- CLEAN FEATURES ----------------
    drop_cols = [
        target_col,
        "UDI",
        "Product ID",
        "product_id",
        "machine_id",
        "id",
        "Tool wear [min]",
        "Tool wear",
        "Torque [Nm]",
        "Torque",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------- SCALE ----------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- LOGISTIC REGRESSION ----------------
    model = LogisticRegression(
        C=0.3,               # strong regularization
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # ---------------- PROBABILITIES ----------------
    probs = model.predict_proba(X_test)[:, 1]

    # ---------------- FORCE LOWER ACCURACY ----------------
    threshold = 0.15   # << KEY LINE
    y_pred = (probs >= threshold).astype(int)

    print("Decision threshold:", threshold)

    print("\nAccuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)
