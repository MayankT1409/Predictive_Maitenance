import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# ---------------- DATASET ----------------
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=5):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx + self.seq_len], dtype=torch.float32),
            torch.tensor(self.y[idx + self.seq_len], dtype=torch.float32),
        )


# ---------------- MODEL ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze()


def main(csv_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(csv_path)
    y = df["Machine failure"].values

    # ---------------- CLEAN FEATURES ----------------
    drop_cols = [
        "Machine failure",
        "UDI",
        "Product ID",
        "Tool wear [min]",
        "Torque [Nm]",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])

    print("Features used:", list(X.columns))

    # ---------------- SCALE ----------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------- SEQUENCES ----------------
    seq_len = 5
    train_ds = SequenceDataset(X_train, y_train, seq_len)
    test_ds = SequenceDataset(X_test, y_test, seq_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    # ---------------- SOFT POS WEIGHT ----------------
    imbalance_ratio = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight = np.sqrt(imbalance_ratio)   # 🔑 KEY FIX
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    print("pos_weight (soft):", round(pos_weight.item(), 2))

    # ---------------- MODEL ----------------
    model = LSTMClassifier(input_dim=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # ---------------- TRAIN ----------------
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | loss {loss.item():.4f}")

    # ---------------- EVAL ----------------
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(yb.numpy())

    probs = np.array(probs)
    labels = np.array(labels)

    # ---------------- THRESHOLD ----------------
    threshold = 0.35
    preds = (probs >= threshold).astype(int)

    print("\nDecision threshold:", threshold)
    print("\nAccuracy:", round(accuracy_score(labels, preds) * 100, 2), "%")
    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, digits=4))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    main(args.csv)
