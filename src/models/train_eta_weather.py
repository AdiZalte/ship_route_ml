"""
train_eta_weather.py
--------------------
ETA prediction with and without weather features.
Copies the logic of train_eta.py but passes real timestamps through
segment_rows so wave_height / wind_speed can be looked up.

Usage:
    python -m src.models.train_eta_weather
"""

from pathlib import Path
import logging

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)

from src.data.weather_join import add_weather, set_train_medians

RAW = Path("data/interim/ais_clean.parquet")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

MAX_VESSELS, MIN_RAW_POINTS = 3000, 1000
INTERVAL_MIN, MAX_GAP_MIN   = 5, 60
ARRIVE_KM, MAX_DIST_KM, MAX_ETA_MIN, STRIDE, SEED = 5, 200, 1440, 3, 42

PORTS = np.array([
    [33.72, -118.27],  # Los Angeles
    [33.75, -118.20],  # Long Beach
    [37.80, -122.39],  # San Francisco
    [47.60, -122.35],  # Seattle
    [29.68, -95.02],   # Houston
    [29.94, -90.06],   # New Orleans
    [25.77, -80.17],   # Miami
    [40.67, -74.05],   # New York
])
PORT_NAMES = ["Los Angeles", "Long Beach", "San Francisco", "Seattle",
              "Houston", "New Orleans", "Miami", "New York"]

FEATS_BASE = ["dist_km", "sog", "sog_mean6", "sin_cog", "cos_cog", "cos_to_port"]
FEATS_W    = FEATS_BASE + ["wave_height", "wind_speed"]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def segment_rows(mmsi, t_min, t_epoch_s, lat, lon, sog, cog):
    """
    t_min      : elapsed minutes (float)
    t_epoch_s  : UTC epoch seconds (int64) – used for weather lookup
    """
    n = len(t_min)
    idx = np.arange(n)
    dist = hav(lat[:, None], lon[:, None], PORTS[None, :, 0], PORTS[None, :, 1])
    near = dist.argmin(1)
    d_near = dist[idx, near]

    nxt = np.full((n, len(PORTS)), n)
    for p in range(len(PORTS)):
        a = np.where(dist[:, p] < ARRIVE_KM, idx, n)
        nxt[:, p] = np.minimum.accumulate(a[::-1])[::-1]
    j = nxt[idx, near]

    ok = ((j < n)
          & (d_near > ARRIVE_KM)
          & (d_near < MAX_DIST_KM)
          & (idx >= 5)
          & (idx % STRIDE == 0))
    m6 = pd.Series(sog).rolling(6).mean().to_numpy()
    ok &= m6 > 3

    out = []
    for i in np.where(ok)[0]:
        eta = t_min[j[i]] - t_min[i]
        if eta > MAX_ETA_MIN:
            continue
        c   = np.radians(cog[i])
        brg = bearing(lat[i], lon[i], PORTS[near[i], 0], PORTS[near[i], 1])
        out.append((
            mmsi,
            lat[i], lon[i],          # for weather lookup
            t_epoch_s[i],             # epoch seconds
            d_near[i], sog[i],
            sog[i - 5:i + 1].mean(),
            np.sin(c), np.cos(c),
            np.cos(c - brg),
            eta,
        ))
    return out


# -- load AIS ------------------------------------------------------------------
con  = duckdb.connect()
src  = str(RAW.resolve()).replace("\\", "/")
print("Loading 5-minute tracks ...")
df = con.execute(f"""
WITH v AS (
    SELECT mmsi FROM read_parquet('{src}')
    WHERE mmsi BETWEEN 100000000 AND 999999999
    GROUP BY mmsi HAVING COUNT(*) >= {MIN_RAW_POINTS}
    ORDER BY hash(mmsi) LIMIT {MAX_VESSELS}
)
SELECT mmsi, timestamp, latitude, longitude, sog, cog FROM (
    SELECT a.mmsi, a.timestamp, a.latitude, a.longitude, a.sog, a.cog,
           ROW_NUMBER() OVER (PARTITION BY a.mmsi,
               time_bucket(INTERVAL '{INTERVAL_MIN} minutes', a.timestamp)
               ORDER BY a.timestamp) AS rn
    FROM read_parquet('{src}') a JOIN v USING (mmsi)
) WHERE rn = 1 ORDER BY mmsi, timestamp
""").df()
con.close()

# -- build rows ----------------------------------------------------------------
rows = []
for mmsi, g in df.groupby("mmsi", sort=False):
    ts_epoch = g["timestamp"].to_numpy().astype("datetime64[s]").astype("int64")
    t_min    = ts_epoch / 60.0
    lat, lon, sog, cog = (g[c].to_numpy(float)
                          for c in ["latitude", "longitude", "sog", "cog"])
    cuts = np.where(np.diff(t_min) > MAX_GAP_MIN)[0] + 1
    for ix in np.split(np.arange(len(g)), cuts):
        if len(ix) >= 12:
            rows += segment_rows(
                mmsi, t_min[ix], ts_epoch[ix],
                lat[ix], lon[ix], sog[ix], cog[ix]
            )

cols = ["mmsi", "lat", "lon", "epoch_s"] + FEATS_BASE + ["eta_min"]
data = pd.DataFrame(rows, columns=cols)
print(f"ETA samples: {len(data):,}  vessels: {data['mmsi'].nunique()}")
if data["mmsi"].nunique() < 10:
    raise RuntimeError("Too few arrivals. Add ports or raise MAX_VESSELS.")

# -- vessel split (same seed / fractions as train_eta.py) ---------------------
ids = data["mmsi"].unique()
np.random.default_rng(SEED).shuffle(ids)
n = len(ids)
train_ids = set(ids[:int(.7 * n)])
val_ids   = set(ids[int(.7 * n):int(.85 * n)])

tr = data[data["mmsi"].isin(train_ids)].copy()
va = data[data["mmsi"].isin(val_ids)].copy()
te = data[~data["mmsi"].isin(train_ids | val_ids)].copy()
print(f"train/val/test: {len(tr)}/{len(va)}/{len(te)}")

# -- weather lookup ------------------------------------------------------------
print("Looking up weather ...")

def add_w(df_split):
    wh, ws = add_weather(df_split["lat"].values,
                         df_split["lon"].values,
                         df_split["epoch_s"].values)
    df_split = df_split.copy()
    df_split["wave_height"] = wh
    df_split["wind_speed"]  = ws
    return df_split

# Compute training medians first
tr_raw_wh, tr_raw_ws = add_weather(tr["lat"].values, tr["lon"].values, tr["epoch_s"].values)
set_train_medians(float(np.nanmedian(tr_raw_wh)), float(np.nanmedian(tr_raw_ws)))

tr = add_w(tr)
va = add_w(va)
te = add_w(te)

nan_w = te[["wave_height", "wind_speed"]].isna().sum()
print(f"  Test NaN: {nan_w.to_dict()}")


# -- metric helpers ------------------------------------------------------------
def report_metrics(name, y, p):
    mae  = mean_absolute_error(y, p)
    rmse = np.sqrt(mean_squared_error(y, p))
    r2   = r2_score(y, p)
    print(f"  {name:<30} MAE {mae:.1f} min | RMSE {rmse:.1f} min | R2 {r2:.3f}")
    return mae, rmse, r2


def baseline_pred(d):
    return np.clip(d["dist_km"] / np.maximum(d["sog"] * 1.852, 0.5) * 60,
                   0, MAX_ETA_MIN)


xgb_params = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=30, n_jobs=-1, random_state=SEED,
)


# -- no-weather model ----------------------------------------------------------
model_base = XGBRegressor(**xgb_params)
model_base.fit(tr[FEATS_BASE], tr["eta_min"],
               eval_set=[(va[FEATS_BASE], va["eta_min"])], verbose=False)

# -- weather model -------------------------------------------------------------
model_w = XGBRegressor(**xgb_params)
model_w.fit(tr[FEATS_W], tr["eta_min"],
            eval_set=[(va[FEATS_W], va["eta_min"])], verbose=False)

# -- results -------------------------------------------------------------------
print("\n" + "="*65)
print("ETA Test-set Results")
print("-"*65)
report_metrics("Baseline (dist/SOG)",     te["eta_min"], baseline_pred(te))
report_metrics("XGBoost (no weather)",    te["eta_min"], model_base.predict(te[FEATS_BASE]))
report_metrics("XGBoost (with weather)",  te["eta_min"], model_w.predict(te[FEATS_W]))
print("="*65)

print("\nFeature importances (XGBoost with weather):")
for feat, imp in sorted(zip(FEATS_W, model_w.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.4f}")

# -- save ----------------------------------------------------------------------
joblib.dump(model_w,    OUT / "eta_weather_xgb.joblib")
joblib.dump(model_base, OUT / "eta_xgb.joblib")      # overwrite with same split for consistency
print(f"\nSaved: {OUT}/eta_weather_xgb.joblib")
