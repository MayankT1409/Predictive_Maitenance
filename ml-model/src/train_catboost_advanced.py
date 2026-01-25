import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from catboost import CatBoostClassifier, Pool
import shap


# ---------------- TARGET DETECTION ----------------
def detect_target_column(df):
    for c in ["Machine failure", "machine_failure", "failure", "Target", "target"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found")


# ---------------- MAIN ----------------
def main(csv_path):
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    print("Detected target:", target_col)

    y = df[target_col]

    drop_cols = [
        target_col,
        "UDI",
        "Product ID",
        "product_id",
        "machine_id",
        "id",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    feature_names = X.columns.tolist()

    print("Features used:", feature_names)
    print("\nTarget distribution:\n", y.value_counts())

    # ---------------- CLASS WEIGHTS ----------------
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weights = [weights[list(classes).index(i)] for i in classes]

    # ---------------- TRAIN / VALIDATION SPLIT ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # ---------------- CATBOOST WITH EARLY STOPPING ----------------
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="F1",
        class_weights=class_weights,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # ---------------- VALIDATION RESULTS ----------------
    y_pred = model.predict(X_val)
    print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
    print("\n=== Validation Classification Report ===")
    print(classification_report(y_val, y_pred, digits=4))

    # ---------------- VALIDATION CURVE ----------------
    evals = model.get_evals_result()
    plt.figure(figsize=(8, 5))
    plt.plot(evals["learn"]["F1"], label="Train F1")
    plt.plot(evals["validation"]["F1"], label="Validation F1")
    plt.xlabel("Iterations")
    plt.ylabel("F1 Score")
    plt.title("CatBoost Training vs Validation Curve")
    plt.legend()
    plt.show()

    # ---------------- SHAP EXPLANATION ----------------
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
        plot_type="bar",
    )

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
    )

    # ---------------- 5-FOLD CROSS VALIDATION ----------------
    print("\nRunning 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="F1",
            class_weights=class_weights,
            random_seed=42,
            verbose=False,
        )

        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        cv_scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    print(
        "\n5-Fold CV Accuracy:",
        f"{np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    main(args.csv)
