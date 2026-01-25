from __future__ import annotations
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)
from sklearn.feature_selection import mutual_info_classif

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.preprocess import preprocess_data
from src.features import add_features


# ================= Dataset =================
class SequenceDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int):
        self.X = X.values.astype("float32")
        self.y = y.values.astype("float32")
        self.seq_len = seq_len
        self.n = len(self.X) - seq_len + 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ================= Model =================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out).squeeze(1)


# ================= Helpers =================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def detect_target(df: pd.DataFrame) -> str:
    if "failure" in df.columns:
        return "failure"
    if "Maintenance Required" in df.columns:
        return "Maintenance Required"
    for c in df.columns:
        if "fail" in c.lower() or "maint" in c.lower():
            return c
    raise ValueError("Target column not found")


def filter_numeric_features(df: pd.DataFrame, target: str) -> list:
    drop = ["fail", "maint", "label", "target", "id", "time"]
    feats = []
    for c in df.columns:
        if c == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(s in c.lower() for s in drop):
            continue
        feats.append(c)
    return feats


def select_top_k_features(X: pd.DataFrame, y: pd.Series, k: int):
    print(f"Selecting top-{k} features using mutual information...")
    mi = mutual_info_classif(X, y, discrete_features=False)
    mi_scores = pd.Series(mi, index=X.columns)
    top_features = mi_scores.sort_values(ascending=False).head(k).index.tolist()
    return top_features


# ================= Training =================
def train_lstm(
    csv_path: str,
    seq_len: int = 32,
    epochs: int = 30,
    batch_size: int = 128,
    hidden_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.3,
    holdout_frac: float = 0.2,
    top_k_features: int = 100,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(csv_path)

    # -------- CHRONOLOGICAL TRAIN SPLIT --------
    split = int(len(df) * (1 - holdout_frac))
    df_train = df.iloc[:split].reset_index(drop=True)
    print("Training rows:", len(df_train))

    # -------- PREPROCESS --------
    df_train = add_features(preprocess_data(df_train), include_trend=False)

    TARGET = detect_target(df_train)
    FEATURES = filter_numeric_features(df_train, TARGET)

    print("Initial feature count:", len(FEATURES))

    # -------- TRAIN / VAL SPLIT --------
    val_frac = 0.1
    tv = int(len(df_train) * (1 - val_frac))
    df_tr = df_train.iloc[:tv]
    df_val = df_train.iloc[tv:]

    X_tr = df_tr[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_tr = df_tr[TARGET].astype("int")

    X_val = df_val[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = df_val[TARGET].astype("int")

    # -------- FEATURE SELECTION --------
    FEATURES = select_top_k_features(X_tr, y_tr, top_k_features)
    print("Reduced feature count:", len(FEATURES))

    X_tr = X_tr[FEATURES]
    X_val = X_val[FEATURES]

    # -------- SCALE --------
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=FEATURES)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=FEATURES)

    # -------- DATASETS --------
    train_ds = SequenceDataset(X_tr_s, y_tr, seq_len)
    val_ds = SequenceDataset(X_val_s, y_val, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -------- MODEL --------
    model = LSTMClassifier(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)

    pos = max(1.0, y_tr.sum())
    neg = max(1.0, (y_tr == 0).sum())
    pos_weight = torch.tensor([neg / pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    patience = 6
    wait = 0

    ensure_dir("models")

    # -------- TRAIN LOOP --------
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # -------- VALIDATION --------
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds.append((probs >= 0.5).cpu().numpy())
                trues.append(yb.numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)

        print(
            f"Epoch {epoch:02d} | "
            f"loss {np.mean(losses):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_f1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(model.state_dict(), "models/lstm_model.pt")
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(FEATURES, "models/features.pkl")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    # -------- FINAL REPORT --------
    print("\n=== FINAL VALIDATION CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, digits=4))

    print("Best validation F1:", best_f1)


# ================= CLI =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--top_k_features", type=int, default=100)
    args = parser.parse_args()

    train_lstm(
        csv_path=args.csv,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        lr=args.lr,
        holdout_frac=args.holdout_frac,
        top_k_features=args.top_k_features,
    )
