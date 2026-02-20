# import pandas as pd
# import numpy as np
# import re

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.utils.class_weight import compute_class_weight

# from xgboost import XGBClassifier


# def clean_feature_names(cols):
#     """
#     XGBoost does not allow [, ], < in feature names
#     """
#     return [
#         re.sub(r"[\[\]<]", "", c).replace(" ", "_")
#         for c in cols
#     ]


# def detect_target_column(df):
#     for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
#         if c in df.columns:
#             return c
#     raise ValueError("Target column not found")


# def main(csv_path):
#     # ---------------- LOAD DATA ----------------
#     df = pd.read_csv(csv_path)

#     target_col = detect_target_column(df)
#     print("Detected target:", target_col)

#     y = df[target_col]
#     print("\nTarget distribution:\n", y.value_counts())

#     # ---------------- REMOVE LEAKAGE ----------------
#     drop_cols = [
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

#     X = df.drop(columns=[c for c in drop_cols if c in df.columns])
#     X = X.select_dtypes(include=[np.number])

#     # ---------------- CLEAN FEATURE NAMES ----------------
#     X.columns = clean_feature_names(X.columns)

#     print("\nFeatures used:")
#     for c in X.columns:
#         print(" -", c)

#     # ---------------- SPLIT ----------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         stratify=y,
#         random_state=42,
#     )

#     # ---------------- CLASS IMBALANCE ----------------
#     classes = np.unique(y_train)
#     weights = compute_class_weight("balanced", classes=classes, y=y_train)
#     class_weight = dict(zip(classes, weights))

#     scale_pos_weight = class_weight[1] / class_weight[0]
#     print("\nscale_pos_weight:", round(scale_pos_weight, 2))

#     # ---------------- XGBOOST (CONSERVATIVE) ----------------
#     model = XGBClassifier(
#         n_estimators=150,          # intentionally small
#         max_depth=3,
#         learning_rate=0.1,
#         subsample=0.7,
#         colsample_bytree=0.7,
#         reg_alpha=1.0,
#         reg_lambda=2.0,
#         scale_pos_weight=scale_pos_weight,
#         eval_metric="logloss",
#         random_state=42,
#         tree_method="hist",        # stable + fast
#     )

#     model.fit(X_train, y_train)

#     # ---------------- EVALUATION ----------------
#     y_pred = model.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     print("\nAccuracy:", round(acc * 100, 2), "%")

#     print("\n=== Classification Report ===")
#     print(classification_report(y_test, y_pred, digits=4))


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True)
#     args = parser.parse_args()

#     main(args.csv)



import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_curve,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier


# ---------------- FEATURE NAME CLEANING ----------------
def clean_feature_names(cols):
    """
    XGBoost does not allow [, ], < in feature names
    """
    return [
        re.sub(r"[\[\]<]", "", c).replace(" ", "_")
        for c in cols
    ]


# ---------------- TARGET DETECTION ----------------
def detect_target_column(df):
    for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found")


# ---------------- MAIN ----------------
def main(csv_path):
    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]
    print("\nTarget distribution:\n", y.value_counts())

    # ---------------- REMOVE LEAKAGE ----------------
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

    # ---------------- CLEAN FEATURE NAMES ----------------
    X.columns = clean_feature_names(X.columns)

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

    # ---------------- CLASS IMBALANCE ----------------
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = dict(zip(classes, weights))

    scale_pos_weight = class_weight[1] / class_weight[0]
    print("\nscale_pos_weight:", round(scale_pos_weight, 2))

    # ---------------- XGBOOST (CONSERVATIVE) ----------------
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------- METRICS ----------------
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", round(acc * 100, 2), "%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # ---------------- ROC CURVE ----------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – XGBoost Model")
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
