import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


def detect_target_column(df):
    for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found")


def main(csv_path):
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]

    # ---------------- CLEAN FEATURE SET ----------------
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

    print("\nFeatures used:")
    for c in X.columns:
        print(" -", c)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------- CLASS WEIGHTS ----------------
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = [weights[list(classes).index(i)] for i in classes]

    # ---------------- CATBOOST ----------------
    model = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=class_weights,
        random_seed=42,
        verbose=False,
    )

    model.fit(X_train, y_train)

    # ---------------- PROBABILITIES ----------------
    y_probs = model.predict_proba(X_test)[:, 1]

    # ---------------- PRECISION–RECALL CURVE ----------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (CatBoost)")
    plt.show()

    # ---------------- THRESHOLD SELECTION ----------------
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print("\nOptimal threshold (max F1):", round(best_threshold, 3))
    print("Precision:", round(precision[best_idx], 3))
    print("Recall:", round(recall[best_idx], 3))
    print("F1:", round(f1_scores[best_idx], 3))

    # ---------------- APPLY NEW THRESHOLD ----------------
    y_pred_opt = (y_probs >= best_threshold).astype(int)

    print("\nAccuracy (threshold tuned):", round(accuracy_score(y_test, y_pred_opt), 4))
    print("\n=== Classification Report (Threshold Tuned) ===")
    print(classification_report(y_test, y_pred_opt, digits=4))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)
