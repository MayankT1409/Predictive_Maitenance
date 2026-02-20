import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


# ---------------- TARGET DETECTION ----------------
def detect_target_column(df):
    for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found")


# ---------------- MAIN ----------------
def main(csv_path):
    # -------- LOAD DATA --------
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]
    print("\nTarget distribution:\n", y.value_counts())

    # -------- REMOVE LEAKAGE --------
    leaky_cols = [
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

    X = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    print("\nFeatures used:")
    for c in X.columns:
        print(" -", c)

    # -------- TRAIN / TEST SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # -------- SCALE FEATURES --------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------- CLASS WEIGHTS --------
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = dict(zip(classes, weights))

    print("\nClass weights:", class_weight)

    # -------- LOGISTIC REGRESSION (GD) --------
    model = LogisticRegression(
        C=0.5,
        max_iter=3000,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -------- PREDICTIONS --------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # -------- METRICS --------
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ======================================================
    # 1️⃣ CONFUSION MATRIX
    # ======================================================
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"]
    )

    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix – Logistic Regression")
    plt.grid(False)
    plt.show()

    # ======================================================
    # 2️⃣ PERFORMANCE BAR GRAPH
    # ======================================================
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [
        acc * 100,
        precision * 100,
        recall * 100,
        f1 * 100,
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, values)
    plt.plot(x, values, linestyle="--", marker="o")

    plt.xticks(x, labels)
    plt.ylabel("In %")
    plt.xlabel("Measures")
    plt.ylim(0, 100)
    plt.title("Logistic Regression (Gradient Descent) Performance")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
        )

    plt.show()

    # ======================================================
    # 3️⃣ ROC CURVE
    # ======================================================
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Gradient Decent (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------- RUN ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)
