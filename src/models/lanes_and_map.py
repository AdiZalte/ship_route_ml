"""
lanes_and_map.py
----------------
1. DBSCAN shipping-lane detection.
2. Predict trajectories with best available LSTM (weather > base).
3. Color actual path red when risk probability > saved threshold.
4. Show wave height, wind speed, ETA and risk probability in marker popups.

Usage:
    python -m src.models.lanes_and_map
"""

import json
import logging
from pathlib import Path

import duckdb
import folium
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree

logging.basicConfig(level=logging.INFO)

RAW = Path("data/interim/ais_clean.parquet")
SEQ = "data/processed/trajectory_sequences.npz"
MODELS, REPORTS = Path("models"), Path("reports")
REPORTS.mkdir(exist_ok=True)

CELL_DEG, SAMPLE_ROWS = 0.02, 1_000_000
EPS_KM, MIN_WEIGHT, OFF_LANE_KM, SEED = 8, 15, 3, 42

PORTS = np.array([
    [33.72, -118.27], [33.75, -118.20], [37.80, -122.39], [47.60, -122.35],
    [29.68, -95.02],  [29.94, -90.06],  [25.77, -80.17],  [40.67, -74.05],
])
PORT_NAMES = ["Los Angeles", "Long Beach", "San Francisco", "Seattle",
              "Houston", "New Orleans", "Miami", "New York"]
FEATS_ETA  = ["dist_km", "sog", "sog_mean6", "sin_cog", "cos_cog", "cos_to_port"]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


# ---------- 1. DBSCAN shipping lanes ----------
print("Sampling moving-ship positions ...")
con = duckdb.connect()
src = str(RAW.resolve()).replace("\\", "/")
pts = con.execute(f"""
SELECT latitude, longitude FROM (
    SELECT latitude, longitude FROM read_parquet('{src}')
    WHERE sog > 3 AND mmsi BETWEEN 100000000 AND 999999999
) USING SAMPLE {SAMPLE_ROWS} ROWS
""").df()
con.close()

cells = (pts.assign(la=(pts["latitude"]  / CELL_DEG).round().astype(int),
                    lo=(pts["longitude"] / CELL_DEG).round().astype(int))
         .groupby(["la", "lo"]).size().reset_index(name="w"))
cells["lat"], cells["lon"] = cells["la"] * CELL_DEG, cells["lo"] * CELL_DEG
xy = np.radians(cells[["lat", "lon"]].to_numpy())

labels = DBSCAN(eps=EPS_KM / 6371, min_samples=MIN_WEIGHT, metric="haversine",
                algorithm="ball_tree").fit(xy, sample_weight=cells["w"].to_numpy()).labels_
cells["label"] = labels
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Cells: {len(cells)} | clusters (lanes): {n_clusters} | noise: {(labels==-1).mean():.1%}")

if n_clusters >= 2:
    m = labels >= 0
    sil = silhouette_score(xy[m], labels[m], metric="haversine",
                           sample_size=min(5000, int(m.sum())), random_state=SEED)
    print(f"Silhouette score: {sil:.3f}")
else:
    print("Fewer than 2 clusters: lower EPS_KM or MIN_WEIGHT.")

lane = cells[cells["label"] >= 0].reset_index(drop=True)
share = lane.groupby("label")["w"].sum().sort_values(ascending=False)
print("Top-5 cluster shares:", (share.head(5) / share.sum()).round(3).tolist())
if lane.empty:
    raise RuntimeError("No lane cells found. Lower MIN_WEIGHT or raise SAMPLE_ROWS.")
tree_lane = BallTree(np.radians(lane[["lat", "lon"]].to_numpy()), metric="haversine")
joblib.dump(lane[["lat", "lon", "label", "w"]], MODELS / "lanes.joblib")


def off_lane_km(lat, lon):
    d, _ = tree_lane.query(np.radians(np.column_stack([lat, lon])))
    return d[:, 0] * 6371


# ---------- 2. LSTM predictions ----------
class LSTMModel(nn.Module):
    def __init__(self, n_feat=5, hidden=64, n_layers=2, out_steps=5):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, out_steps * 2))
        self._out = out_steps

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).reshape(-1, self._out, 2)


seq_data   = np.load(SEQ)
X, Y       = seq_data["X_test"], seq_data["y_test"]
T_test     = seq_data["T_test"] if "T_test" in seq_data else None
N_STEPS    = X.shape[1]   # 20
OUTPUT_STEPS = Y.shape[1]  # 5

# Prefer weather LSTM if available
use_weather_lstm = (MODELS / "lstm_weather.pt").exists() and \
                   (MODELS / "scalers_weather.joblib").exists()

if use_weather_lstm:
    print("Using weather LSTM ...")
    from src.data.weather_join import add_weather, set_train_medians
    sc = joblib.load(MODELS / "scalers_weather.joblib")
    N_FEAT = sc["x"].n_features_in_   # 7
    model_lstm = LSTMModel(n_feat=N_FEAT)
    model_lstm.load_state_dict(
        torch.load(MODELS / "lstm_weather.pt", map_location="cpu"))

    # build weather features
    if T_test is not None:
        try:
            wh, ws = add_weather(X[:, -1, 0], X[:, -1, 1], T_test)
            set_train_medians(float(np.nanmedian(wh)), float(np.nanmedian(ws)))
            wh, ws = add_weather(X[:, -1, 0], X[:, -1, 1], T_test)
        except FileNotFoundError:
            wh = np.zeros(len(X), dtype=np.float32)
            ws = np.zeros(len(X), dtype=np.float32)
    else:
        wh = np.zeros(len(X), dtype=np.float32)
        ws = np.zeros(len(X), dtype=np.float32)

    Wb = np.column_stack([wh, ws])[:, np.newaxis, :].repeat(N_STEPS, axis=1)
    X7 = np.concatenate([X, Wb], axis=2).astype(np.float32)
    xs = sc["x"].transform(X7.reshape(-1, N_FEAT)).reshape(-1, N_STEPS, N_FEAT)
    lstm_label = "LSTM+weather predicted"
else:
    print("Using base LSTM ...")
    sc = joblib.load(MODELS / "scalers.joblib")
    model_lstm = LSTMModel(n_feat=5)
    model_lstm.load_state_dict(
        torch.load(MODELS / "lstm.pt", map_location="cpu"))
    xs = sc["x"].transform(X.reshape(-1, 5)).reshape(-1, N_STEPS, 5)
    wh = np.zeros(len(X), dtype=np.float32)
    ws = np.zeros(len(X), dtype=np.float32)
    lstm_label = "LSTM predicted"

model_lstm.eval()
with torch.no_grad():
    p = model_lstm(torch.tensor(xs, dtype=torch.float32)).numpy()
P = sc["y"].inverse_transform(p.reshape(-1, 2)).reshape(-1, OUTPUT_STEPS, 2) + X[:, -1:, :2]

last = X[:, -1, :2].astype(float)
dkm  = off_lane_km(last[:, 0], last[:, 1])
for t in (2, 3, 5, 10):
    print(f"Test windows off-lane (> {t} km): {(dkm > t).mean():.1%}")
off = dkm > OFF_LANE_KM
print(f"Test windows off-lane: {off.mean():.1%}")

# ---------- 3. Risk probabilities ----------
risk_prob = np.zeros(len(X), dtype=np.float32)
risk_threshold = 0.5

risk_model_path = MODELS / "risk_xgb.joblib"
risk_thr_path   = MODELS / "risk_threshold.json"

if risk_model_path.exists() and risk_thr_path.exists():
    print("Loading risk classifier ...")
    clf_risk = joblib.load(risk_model_path)
    with open(risk_thr_path) as f:
        thr_data = json.load(f)
    risk_threshold = thr_data["threshold"]

    F_risk = np.column_stack([
        last[:, 0], last[:, 1],
        X[:, -1, 2],
        X[:, :, 2].mean(1),
        X[:, -1, 3], X[:, -1, 4],
        wh, ws,
    ]).astype(np.float32)
    # Only pass as many features as the model was trained on
    n_risk_feat = clf_risk.n_features_in_
    risk_prob = clf_risk.predict_proba(F_risk[:, :n_risk_feat])[:, 1].astype(np.float32)
    high_risk = (risk_prob >= risk_threshold)
    print(f"High-risk windows: {high_risk.mean():.1%} (threshold={risk_threshold:.2f})")
else:
    print("Risk model not found; skipping risk coloring.")
    high_risk = np.zeros(len(X), dtype=bool)

# ---------- 4. ETA ----------
# Prefer weather ETA if available
use_weather_eta = (MODELS / "eta_weather_xgb.joblib").exists()
if use_weather_eta:
    print("Using weather ETA model ...")
    eta_model = joblib.load(MODELS / "eta_weather_xgb.joblib")
    eta_n_feat = eta_model.n_features_in_
else:
    eta_model  = joblib.load(MODELS / "eta_xgb.joblib")
    eta_n_feat = eta_model.n_features_in_
    print("Using base ETA model ...")

dist   = hav(last[:, None, 0], last[:, None, 1], PORTS[None, :, 0], PORTS[None, :, 1])
near   = dist.argmin(1)
d_near = dist[np.arange(len(X)), near]
course = np.arctan2(X[:, -1, 3], X[:, -1, 4])
brg    = bearing(last[:, 0], last[:, 1], PORTS[near, 0], PORTS[near, 1])

F_eta = pd.DataFrame({
    "dist_km":    d_near,
    "sog":        X[:, -1, 2],
    "sog_mean6":  X[:, -6:, 2].mean(1),
    "sin_cog":    X[:, -1, 3],
    "cos_cog":    X[:, -1, 4],
    "cos_to_port": np.cos(course - brg),
    "wave_height": wh,
    "wind_speed":  ws,
})
# Slice to however many features the model needs
eta_min = eta_model.predict(F_eta.iloc[:, :eta_n_feat])

# ---------- 5. Map ----------
rng  = np.random.default_rng(SEED)
pool = np.where((d_near > 5) & (d_near < 200))[0]
f    = int(rng.choice(pool if len(pool) else np.arange(len(X))))
idx  = np.where(hav(last[:, 0], last[:, 1], last[f, 0], last[f, 1]) < 40)[0][:40]

mp = folium.Map(location=[last[f, 0], last[f, 1]], zoom_start=10, tiles="OpenStreetMap")

near_cells = lane[hav(lane["lat"].values, lane["lon"].values, last[f, 0], last[f, 1]) < 80]
for r in near_cells.itertuples():
    folium.CircleMarker([r.lat, r.lon], radius=2, color="#888", weight=0,
                        fill=True, fill_opacity=0.35).add_to(mp)

for i in idx:
    is_high_risk = bool(high_risk[i])
    # Past track (blue)
    folium.PolyLine(X[i, :, :2].astype(float).tolist(),
                    color="#1f77b4", weight=2).add_to(mp)
    # Actual future track: red if high-risk, green otherwise
    actual_color = "#d62728" if is_high_risk else "#2ca02c"
    folium.PolyLine(np.vstack([X[i, -1, :2], Y[i]]).astype(float).tolist(),
                    color=actual_color, weight=3, tooltip="actual").add_to(mp)
    # LSTM predicted track (orange dashed)
    folium.PolyLine(np.vstack([X[i, -1, :2], P[i]]).astype(float).tolist(),
                    color="#ff7f0e", weight=3, dash_array="6",
                    tooltip=lstm_label).add_to(mp)

    if 5 < d_near[i] < 200:
        txt = (
            f"<b>ETA to {PORT_NAMES[near[i]]}</b>: {eta_min[i]/60:.1f} h"
            f"<br>{d_near[i]:.0f} km away"
            f"<br>Wave: {wh[i]:.1f} m | Wind: {ws[i]:.1f} m/s"
            f"<br>Risk prob: {risk_prob[i]:.2f}"
        )
        if is_high_risk:
            txt += "<br><b style='color:red'>! High deviation risk</b>"
        if off[i]:
            txt += "<br><b>Off-lane</b>"
        folium.CircleMarker(last[i].tolist(), radius=4,
                            color="#d62728" if is_high_risk else "black",
                            fill=True, fill_opacity=0.8,
                            popup=folium.Popup(txt, max_width=240)).add_to(mp)

legend = (
    "<div style='position:fixed;bottom:20px;left:20px;z-index:9999;"
    "background:white;padding:10px;border:1px solid #999;font-size:12px;border-radius:4px'>"
    "<b>Legend</b><br>"
    "<span style='color:#1f77b4'>&#9632;</span> Past track<br>"
    "<span style='color:#2ca02c'>&#9632;</span> Actual next 25 min<br>"
    "<span style='color:#d62728'>&#9632;</span> Actual -- high risk / off-lane<br>"
    f"<span style='color:#ff7f0e'>&#9632;</span> {lstm_label}<br>"
    "<span style='color:#888'>&#9679;</span> Lane cells<br>"
    f"<i>Risk threshold: {risk_threshold:.2f}</i>"
    "</div>"
)
mp.get_root().html.add_child(folium.Element(legend))
mp.save(REPORTS / "route_map.html")
print("Saved:", REPORTS / "route_map.html")