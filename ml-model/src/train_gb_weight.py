import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def detect_target_column(df):
    candidates = [
        "Machine failure",
        "machine_failure",
        "failure",
        "target",
        "Target",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No target column found. Columns: {list(df.columns)}")


def main(csv_path):
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(csv_path)

    # ---------------- TARGET ----------------
    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]

    print("\nTarget distribution:")
    print(y.value_counts())

    # ---------------- FEATURES ----------------
    drop_cols = [
        target_col,
        "UDI",
        "Product ID",
        "product_id",
        "id",
        "machine_id",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    print("\nNumber of features used:", X.shape[1])

    # ---------------- SCALE ----------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------- CLASS WEIGHTS ----------------
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = dict(zip(classes, weights))

    print("\nComputed class weights:", class_weight)

    # ---------------- MODEL ----------------
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    # GradientBoosting does not natively accept class_weight,
    # so we apply it via sample_weight
    sample_weight = y_train.map(class_weight)

    gb.fit(X_train, y_train, sample_weight=sample_weight)

    # ---------------- EVALUATION ----------------
    y_pred = gb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc * 100, 2), "%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    args = parser.parse_args()

    main(args.csv)
