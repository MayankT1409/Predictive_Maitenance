import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import shap

from .preprocess import preprocess_data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for predictive maintenance"""

    # Rolling statistics
    df["temp_roll_mean5"] = df["temperature"].rolling(window=5, min_periods=1).mean()
    df["vib_roll_std5"] = df["vibration"].rolling(window=5, min_periods=1).std().fillna(0)

    df["temp_roll_max10"] = df["temperature"].rolling(window=10, min_periods=1).max()
    df["vib_roll_mean10"] = df["vibration"].rolling(window=10, min_periods=1).mean()
    df["rpm_roll_std10"] = df["rpm"].rolling(window=10, min_periods=1).std().fillna(0)

    # Lag features
    df["temp_lag1"] = df["temperature"].shift(1).fillna(df["temperature"].iloc[0])
    df["rpm_lag1"] = df["rpm"].shift(1).fillna(df["rpm"].iloc[0])

    # Change rates
    df["rpm_change"] = df["rpm"].diff().fillna(0)
    df["temp_change"] = df["temperature"].diff().fillna(0)
    df["vib_change"] = df["vibration"].diff().fillna(0)

    # Interaction features
    df["vib_rpm"] = df["vibration"] * df["rpm"]
    df["press_temp_ratio"] = df["pressure"] / (df["temperature"] + 1e-6)

    return df


def train_model(csv_path: str, model_path: str = "models/gb_model.pkl"):
    # Load dataset
    df = pd.read_csv(csv_path)
    print("Columns:", df.columns.tolist())

    # Preprocess
    df = preprocess_data(df)

    # Add engineered features
    df = add_features(df)

    # Drop constant / duplicate columns
    df = df.loc[:, df.nunique() > 1]

    # Features & target
    X = df.drop("failure", axis=1)
    y = df["failure"]

    print("\nClass Distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Define model (Gradient Boosting)
    gb = GradientBoostingClassifier(random_state=42)

    # Hyperparameter search space
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
    }

    # Time series split for predictive maintenance
    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        gb, param_distributions=param_dist,
        n_iter=20, scoring="f1",
        cv=tscv, verbose=1, random_state=42, n_jobs=-1
    )

    # Fit model
    search.fit(X_scaled, y)
    model = search.best_estimator_
    print(f"Best Params: {search.best_params_}")

    # Predictions (using last 20% for evaluation)
    split_index = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-15:]
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
    plt.title("Feature Importance (Gradient Boosting)")
    plt.show()

    # SHAP values for feature explainability
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved at {model_path}")

    return model, X_test, y_test


