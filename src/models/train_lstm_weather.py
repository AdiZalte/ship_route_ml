"""
train_lstm_weather.py
---------------------
Same 2-layer LSTM as train_lstm.py but with 7 input features:
  lat, lon, sog, sin_cog, cos_cog, wave_height, wind_speed

Wave height and wind speed are looked up for the last input step of each
window (from the T_train/T_val/T_test timestamps in trajectory_sequences.npz)
and broadcast across all 20 timesteps.

Usage:
    python -m src.models.train_lstm_weather
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import logging
logging.basicConfig(level=logging.INFO)

from src.data.weather_join import add_weather, set_train_medians

DATA   = "data/processed/trajectory_sequences.npz"
OUT    = "models"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -- load sequences ----------------------------------------------------------
data = np.load(DATA)
X_train, y_train, T_train = data["X_train"], data["y_train"], data["T_train"]
X_val,   y_val,   T_val   = data["X_val"],   data["y_val"],   data["T_val"]
X_test,  y_test,  T_test  = data["X_test"],  data["y_test"],  data["T_test"]

INPUT_STEPS  = X_train.shape[1]   # 20
OUTPUT_STEPS = y_train.shape[1]   # 5
N_BASE_FEAT  = X_train.shape[2]   # 5
N_FEAT       = N_BASE_FEAT + 2    # +wave, +wind = 7


def _get_weather(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Return (N, 2) array [wave_height, wind_speed] for each window."""
    lat = X[:, -1, 0].astype(float)
    lon = X[:, -1, 1].astype(float)
    wh, ws = add_weather(lat, lon, T)
    return np.column_stack([wh, ws]).astype(np.float32)


# -- weather lookup -----------------------------------------------------------
print("Looking up weather for train/val/test ...")
W_train = _get_weather(X_train, T_train)
W_val   = _get_weather(X_val,   T_val)
W_test  = _get_weather(X_test,  T_test)

# Set medians from training split for NaN fill
train_wave_med = float(np.nanmedian(W_train[:, 0]))
train_wind_med = float(np.nanmedian(W_train[:, 1]))
set_train_medians(train_wave_med, train_wind_med)
# Re-lookup so medians are applied
W_train = _get_weather(X_train, T_train)
W_val   = _get_weather(X_val,   T_val)
W_test  = _get_weather(X_test,  T_test)

print(f"  Train weather NaN remaining: {np.isnan(W_train).sum()}")


def _augment(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Broadcast per-window weather over timesteps -> (N, T, 7)."""
    N, T, _ = X.shape
    Wb = W[:, np.newaxis, :].repeat(T, axis=1)   # (N, T, 2)
    return np.concatenate([X, Wb], axis=2).astype(np.float32)


X_train7 = _augment(X_train, W_train)
X_val7   = _augment(X_val,   W_val)
X_test7  = _augment(X_test,  W_test)


# -- scalers ------------------------------------------------------------------
def displacement(X, Y):
    return Y - X[:, -1:, :2]

d_train = displacement(X_train, y_train)
d_val   = displacement(X_val,   y_val)

x_scaler = StandardScaler().fit(X_train7.reshape(-1, N_FEAT))
y_scaler = StandardScaler().fit(d_train.reshape(-1, 2))


def make_loader(X7, X, Y, shuffle=False):
    xs  = x_scaler.transform(X7.reshape(-1, N_FEAT)).reshape(-1, INPUT_STEPS, N_FEAT)
    ds  = y_scaler.transform(displacement(X, Y).reshape(-1, 2)).reshape(-1, OUTPUT_STEPS, 2)
    last = X[:, -1, :2]
    ds = TensorDataset(
        torch.tensor(xs,   dtype=torch.float32),
        torch.tensor(ds,   dtype=torch.float32),
        torch.tensor(last, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=512, shuffle=shuffle)


train_loader = make_loader(X_train7, X_train, y_train, True)
val_loader   = make_loader(X_val7,   X_val,   y_val)


# -- model --------------------------------------------------------------------
class LSTMWeatherModel(nn.Module):
    def __init__(self, n_feat: int = N_FEAT, hidden: int = 64,
                 n_layers: int = 2, out_steps: int = OUTPUT_STEPS):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_steps * 2),
        )
        self._out_steps = out_steps

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).reshape(-1, self._out_steps, 2)


def haversine_loss(pred, target, last):
    scale = torch.tensor(y_scaler.scale_, dtype=pred.dtype, device=pred.device)
    mean  = torch.tensor(y_scaler.mean_,  dtype=pred.dtype, device=pred.device)
    pred   = pred   * scale + mean
    target = target * scale + mean
    dlat = torch.deg2rad(pred[..., 0] - target[..., 0])
    dlon = torch.deg2rad(pred[..., 1] - target[..., 1])
    lat1 = torch.deg2rad(last[:, 0:1] + target[..., 0])
    lat2 = torch.deg2rad(last[:, 0:1] + pred[..., 0])
    a = (torch.sin(dlat / 2).square()
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).square()).clamp(0, 1)
    return (6371 * 2 * torch.asin(torch.sqrt(a + 1e-12))).mean()


model     = LSTMWeatherModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss, best_state, wait = float("inf"), None, 0

for epoch in range(30):
    model.train()
    total = 0
    for X, y, last in train_loader:
        X, y, last = X.to(device), y.to(device), last.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(X)

    model.eval()
    val = 0.0
    with torch.no_grad():
        for X, y, last in val_loader:
            X, y, last = X.to(device), y.to(device), last.to(device)
            val += haversine_loss(model(X), y, last).item() * len(X)
    val /= len(val_loader.dataset)
    print(f"epoch {epoch+1:2d} | train_mse {total/len(train_loader.dataset):.4f} | val_km {val:.3f}")

    if val < best_loss:
        best_loss, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= 5:
            break


# -- save ---------------------------------------------------------------------
torch.save(best_state, f"{OUT}/lstm_weather.pt")
joblib.dump({"x": x_scaler, "y": y_scaler}, f"{OUT}/scalers_weather.joblib")


# -- evaluate -----------------------------------------------------------------
def haversine_np(actual, pred):
    a = np.radians(actual)
    b = np.radians(pred)
    h = (np.sin((b[..., 0] - a[..., 0]) / 2) ** 2
         + np.cos(a[..., 0]) * np.cos(b[..., 0])
         * np.sin((b[..., 1] - a[..., 1]) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


model.load_state_dict(best_state)
model.eval()
xs_test = x_scaler.transform(X_test7.reshape(-1, N_FEAT)).reshape(-1, INPUT_STEPS, N_FEAT)
with torch.no_grad():
    p = model(torch.tensor(xs_test, dtype=torch.float32).to(device)).cpu().numpy()
P = y_scaler.inverse_transform(p.reshape(-1, 2)).reshape(-1, OUTPUT_STEPS, 2) + X_test[:, -1:, :2]
err_w = haversine_np(y_test, P)

# -- baselines (recomputed here for comparison) --------------------------------
stationary = np.repeat(X_test[:, -1:, :2], OUTPUT_STEPS, axis=1)
err_stat = haversine_np(y_test, stationary)

sog = X_test[:, -1, 2] * 1.852 / 60     # m/min in km/min
course = np.arctan2(X_test[:, -1, 3], X_test[:, -1, 4])
cs_preds = []
for minutes in range(5, OUTPUT_STEPS * 5 + 1, 5):
    dist = sog * minutes
    lat1 = np.radians(X_test[:, -1, 0])
    lon1 = np.radians(X_test[:, -1, 1])
    ad   = dist / 6371
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ad)
                     + np.cos(lat1) * np.sin(ad) * np.cos(course))
    lon2 = lon1 + np.arctan2(np.sin(course) * np.sin(ad) * np.cos(lat1),
                              np.cos(ad) - np.sin(lat1) * np.sin(lat2))
    cs_preds.append(np.column_stack([np.degrees(lat2),
                                     (np.degrees(lon2) + 540) % 360 - 180]))
cs_pred_arr = np.stack(cs_preds, axis=1)
err_cs = haversine_np(y_test, cs_pred_arr)

# Load plain LSTM results if available
lstm_pt = f"{OUT}/lstm.pt"
scalers_base = f"{OUT}/scalers.joblib"
if os.path.exists(lstm_pt) and os.path.exists(scalers_base):
    # Re-define LSTMModel locally so we don't re-run train_lstm.py on import
    class _LSTMBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1]).reshape(-1, 5, 2)

    sc_base = joblib.load(scalers_base)
    m_base  = _LSTMBase()
    m_base.load_state_dict(torch.load(lstm_pt, map_location="cpu"))
    m_base.eval()
    xs_b = sc_base["x"].transform(X_test.reshape(-1, 5)).reshape(-1, 20, 5)
    with torch.no_grad():
        pb = m_base(torch.tensor(xs_b, dtype=torch.float32)).numpy()
    Pb = sc_base["y"].inverse_transform(pb.reshape(-1, 2)).reshape(-1, OUTPUT_STEPS, 2) + X_test[:, -1:, :2]
    err_base = haversine_np(y_test, Pb)
    base_ade = err_base.mean()
    base_fde = err_base[:, -1].mean()
else:
    base_ade = base_fde = float("nan")

# -- comparison table ---------------------------------------------------------
print("\n" + "="*60)
print(f"{'Model':<30} {'ADE (km)':>10} {'FDE (km)':>10}")
print("-"*60)
print(f"{'Stationary':<30} {err_stat.mean():>10.3f} {err_stat[:,-1].mean():>10.3f}")
print(f"{'Course-speed':<30} {err_cs.mean():>10.3f} {err_cs[:,-1].mean():>10.3f}")
print(f"{'LSTM (no weather)':<30} {base_ade:>10.3f} {base_fde:>10.3f}")
print(f"{'LSTM (with weather)':<30} {err_w.mean():>10.3f} {err_w[:,-1].mean():>10.3f}")
print("="*60)
if err_w.mean() <= base_ade:
    print("Weather features IMPROVED trajectory prediction.")
else:
    print("Weather features did NOT improve trajectory prediction "
          "(weather is current reanalysis, not a forecast; "
          "displacement is dominated by inertia at 5-min steps).")
print(f"\nSaved: {OUT}/lstm_weather.pt  {OUT}/scalers_weather.joblib")
