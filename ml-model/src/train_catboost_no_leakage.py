import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


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
    print("\nTarget distribution:\n", y.value_counts())

    # ---------------- DROP ALL LEAKY COLUMNS ----------------
    leaky_cols = [
        target_col,
        "UDI",
        "Product ID",
        "product_id",
        "machine_id",
        "id",
        # direct causal
        "Tool wear [min]",
        "Tool wear",
        "Torque [Nm]",
        "Torque",
        # failure labels (CRITICAL)
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    print("\nFeatures used:")
    for c in X.columns:
        print(" -", c)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ---------------- CLASS WEIGHTS ----------------
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = [weights[list(classes).index(i)] for i in classes]

    print("\nClass weights:", dict(zip(classes, weights)))

    # ---------------- CATBOOST ----------------
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
    )

    model.fit(X_train, y_train)

    # ---------------- EVAL ----------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc * 100, 2), "%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)
