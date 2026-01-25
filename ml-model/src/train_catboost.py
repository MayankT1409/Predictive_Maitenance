import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


def detect_target_column(df):
    candidates = [
        "Machine failure",
        "machine_failure",
        "failure",
        "Target",
        "target",
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

    # ---------------- TRAIN / TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
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
    class_weights = dict(zip(classes, weights))

    print("\nComputed class weights:", class_weights)

    # CatBoost expects list ordered by class index
    cat_class_weights = [class_weights[c] for c in sorted(class_weights.keys())]

    # ---------------- MODEL ----------------
    model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="F1",
    class_weights=cat_class_weights,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
)

    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    y_pred = model.predict(X_test)

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
