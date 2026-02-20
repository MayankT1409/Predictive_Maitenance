# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )
# from sklearn.utils.class_weight import compute_class_weight

# from catboost import CatBoostClassifier


# # ---------------- TARGET DETECTION ----------------
# def detect_target_column(df):
#     for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
#         if c in df.columns:
#             return c
#     raise ValueError("Target column not found")


# # ---------------- MAIN ----------------
# def main(csv_path):
#     # -------- LOAD DATA --------
#     df = pd.read_csv(csv_path)

#     target_col = detect_target_column(df)
#     print("Detected target:", target_col)

#     y = df[target_col]
#     print("\nTarget distribution:\n", y.value_counts())

#     # -------- REMOVE LEAKAGE --------
#     leaky_cols = [
#         target_col,
#         "UDI",
#         "Product ID",
#         "product_id",
#         "machine_id",
#         "id",
#         "Tool wear [min]",
#         "Tool wear",
#         "Torque [Nm]",
#         "Torque",
#         "TWF",
#         "HDF",
#         "PWF",
#         "OSF",
#         "RNF",
#     ]

#     X = df.drop(columns=[c for c in leaky_cols if c in df.columns])
#     X = X.select_dtypes(include=[np.number])

#     print("\nFeatures used:")
#     for c in X.columns:
#         print(" -", c)

#     # -------- TRAIN / TEST SPLIT --------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         stratify=y,
#         random_state=42,
#     )

#     # -------- CLASS WEIGHTS --------
#     classes = np.unique(y_train)
#     weights = compute_class_weight(
#         class_weight="balanced",
#         classes=classes,
#         y=y_train,
#     )
#     class_weights = [weights[list(classes).index(i)] for i in classes]

#     print("\nClass weights:", dict(zip(classes, weights)))

#     # -------- CATBOOST MODEL --------
#     model = CatBoostClassifier(
#         iterations=600,
#         depth=6,
#         learning_rate=0.05,
#         loss_function="Logloss",
#         eval_metric="F1",
#         class_weights=class_weights,
#         random_seed=42,
#         verbose=100,
#     )

#     model.fit(X_train, y_train)

#     # -------- PREDICTION --------
#     y_pred = model.predict(X_test)

#     # -------- METRICS --------
#     acc = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print("\nAccuracy:", round(acc * 100, 2), "%")
#     print("\n=== Classification Report ===")
#     print(classification_report(y_test, y_pred, digits=4))

#     # ======================================================
#     # 1️⃣ CONFUSION MATRIX
#     # ======================================================
#     cm = confusion_matrix(y_test, y_pred)

#     disp = ConfusionMatrixDisplay(
#         confusion_matrix=cm,
#         display_labels=["No Failure", "Failure"]
#     )

#     plt.figure(figsize=(6, 5))
#     disp.plot(cmap="Blues", values_format="d")
#     plt.title("Confusion Matrix for CatBoost Model")
#     plt.grid(False)
#     plt.show()

#     # ======================================================
#     # 2️⃣ PERFORMANCE BAR GRAPH (REFERENCE STYLE)
#     # ======================================================
#     labels = ["Accuracy", "Precision", "Recall", "F1-score"]
#     values = [
#         acc * 100,
#         precision * 100,
#         recall * 100,
#         f1 * 100,
#     ]

#     x = np.arange(len(labels))

#     plt.figure(figsize=(8, 5))
#     bars = plt.bar(x, values)

#     # Dotted trend line
#     plt.plot(x, values, linestyle="--", marker="o")

#     plt.xticks(x, labels)
#     plt.ylabel("In %")
#     plt.xlabel("Measures")
#     plt.ylim(75, 100)
#     plt.title("Gradient Boost Model Performance")

#     # Value labels on bars
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             yval + 0.5,
#             f"{yval:.2f}",
#             ha="center",
#             va="bottom",
#             fontsize=9,
#         )

#     plt.show()


# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True)
#     args = parser.parse_args()

#     main(args.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier


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

    # -------- SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # -------- CLASS WEIGHTS --------
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = [weights[list(classes).index(i)] for i in classes]

    print("\nClass weights:", dict(zip(classes, weights)))

    # ======================================================
    # 🔥 CATBOOST – ROC OPTIMIZED (DEPTH INCREASED)
    # ======================================================
    model = CatBoostClassifier(
        iterations=800,          # more trees → smoother ranking
        depth=7,                 # 🔑 increased from 6 → better ROC
        learning_rate=0.03,      # lower LR for better probability ordering
        loss_function="Logloss",
        eval_metric="AUC",       # 🔑 optimize ROC directly
        class_weights=class_weights,
        random_seed=42,
        verbose=100,
    )

    model.fit(X_train, y_train)

    # -------- PREDICTIONS --------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # -------- METRICS --------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print("ROC-AUC:", round(auc, 4))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # -------- ROC CURVE --------
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"CatBoost-AUC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – CatBoost (Depth = 7, AUC Optimized)")
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
