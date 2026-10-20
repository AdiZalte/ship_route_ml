import os
import copy
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA = "data/processed/trajectory_sequences.npz"
OUT = "models"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data = np.load(DATA)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

def displacement(X, Y):
    return Y - X[:, -1:, :2]

d_train = displacement(X_train, y_train)
d_val = displacement(X_val, y_val)

x_scaler = StandardScaler().fit(X_train.reshape(-1, 5))
y_scaler = StandardScaler().fit(d_train.reshape(-1, 2))

def make_loader(X, Y, shuffle=False):
    xs = x_scaler.transform(X.reshape(-1, 5)).reshape(-1, 20, 5)
    ds = y_scaler.transform(displacement(X, Y).reshape(-1, 2)).reshape(-1, 5, 2)
    last = X[:, -1, :2]

    dataset = TensorDataset(
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ds, dtype=torch.float32),
        torch.tensor(last, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=512, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, True)
val_loader = make_loader(X_val, y_val)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).reshape(-1, 5, 2)

def haversine_loss(pred, target, last):
    scale = torch.tensor(y_scaler.scale_, dtype=pred.dtype, device=pred.device)
    mean = torch.tensor(y_scaler.mean_, dtype=pred.dtype, device=pred.device)

    pred = pred * scale + mean
    target = target * scale + mean

    lat1 = torch.deg2rad(last[:, 0:1] + target[..., 0])
    lon1 = torch.deg2rad(last[:, 1:2] + target[..., 1])
    lat2 = torch.deg2rad(last[:, 0:1] + pred[..., 0])
    lon2 = torch.deg2rad(last[:, 1:2] + pred[..., 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        torch.sin(dlat / 2).square()
        + torch.cos(lat1) * torch.cos(lat2)
        * torch.sin(dlon / 2).square()
    ).clamp(0, 1)

    return (6371 * 2 * torch.asin(torch.sqrt(a + 1e-7))).mean()

model = LSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss = float("inf")
best_state = None
wait = 0

for epoch in range(30):
    model.train()
    total = 0

    for X, y, last in train_loader:
        X, y, last = X.to(device), y.to(device), last.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)