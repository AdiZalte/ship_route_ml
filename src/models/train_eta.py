from pathlib import Path
import duckdb, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RAW = Path("data/interim/ais_clean.parquet")
OUT = Path("models"); OUT.mkdir(exist_ok=True)
MAX_VESSELS, MIN_RAW_POINTS = 3000, 1000
INTERVAL_MIN, MAX_GAP_MIN = 5, 60
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
FEATS = ["dist_km", "sog", "sog_mean6", "sin_cog", "cos_cog", "cos_to_port"]


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def segment_rows(mmsi, t, lat, lon, sog, cog):
    n = len(t); idx = np.arange(n)
    dist = hav(lat[:, None], lon[:, None], PORTS[None, :, 0], PORTS[None, :, 1])
    near = dist.argmin(1); d_near = dist[idx, near]
    nxt = np.full((n, len(PORTS)), n)
    for p in range(len(PORTS)):
        a = np.where(dist[:, p] < ARRIVE_KM, idx, n)
        nxt[:, p] = np.minimum.accumulate(a[::-1])[::-1]   # next arrival index
    j = nxt[idx, near]
    ok = (j < n) & (d_near > ARRIVE_KM) & (d_near < MAX_DIST_KM) & (idx >= 5) & (idx % STRIDE == 0)
    m6 = pd.Series(sog).rolling(6).mean().to_numpy()
    ok &= m6 > 3          
    out = []
    for i in np.where(ok)[0]:
        eta = t[j[i]] - t[i]
        if eta > MAX_ETA_MIN:
            continue
        c = np.radians(cog[i])
        brg = bearing(lat[i], lon[i], PORTS[near[i], 0], PORTS[near[i], 1])
        out.append((mmsi, d_near[i], sog[i], sog[i - 5:i + 1].mean(),
                    np.sin(c), np.cos(c), np.cos(c - brg), eta))
    return out


con = duckdb.connect()
src = str(RAW.resolve()).replace("\\", "/")
print("Loading 5-minute tracks...")
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

rows = []
for mmsi, g in df.groupby("mmsi", sort=False):
    t = g["timestamp"].to_numpy().astype("datetime64[s]").astype("int64") / 60.0
    lat, lon, sog, cog = (g[c].to_numpy(float) for c in ["latitude", "longitude", "sog", "cog"])
    cuts = np.where(np.diff(t) > MAX_GAP_MIN)[0] + 1
    for ix in np.split(np.arange(len(g)), cuts):
        if len(ix) >= 12:
            rows += segment_rows(mmsi, t[ix], lat[ix], lon[ix], sog[ix], cog[ix])

data = pd.DataFrame(rows, columns=["mmsi"] + FEATS + ["eta_min"])
print("ETA samples:", len(data), "vessels:", data["mmsi"].nunique())
if data["mmsi"].nunique() < 10:
    raise RuntimeError("Too few arrivals. Add ports to PORTS or raise MAX_VESSELS.")

ids = data["mmsi"].unique()
np.random.default_rng(SEED).shuffle(ids)
n = len(ids)
train_ids, val_ids = set(ids[:int(.7 * n)]), set(ids[int(.7 * n):int(.85 * n)])
tr = data[data["mmsi"].isin(train_ids)]
va = data[data["mmsi"].isin(val_ids)]
te = data[~data["mmsi"].isin(train_ids | val_ids)]
print("train/val/test:", len(tr), len(va), len(te))


def report(name, y, p):
    print(f"{name}: MAE {mean_absolute_error(y, p):.1f} min | "
          f"RMSE {np.sqrt(mean_squared_error(y, p)):.1f} min | R2 {r2_score(y, p):.3f}")


def baseline(d):  # distance-to-go / current speed
    return np.clip(d["dist_km"] / np.maximum(d["sog"] * 1.852, 0.5) * 60, 0, MAX_ETA_MIN)


model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, early_stopping_rounds=30,
                     n_jobs=-1, random_state=SEED)
model.fit(tr[FEATS], tr["eta_min"], eval_set=[(va[FEATS], va["eta_min"])], verbose=False)

report("Baseline dist/SOG", te["eta_min"], baseline(te))
report("XGBoost         ", te["eta_min"], model.predict(te[FEATS]))
print("Feature importance:")
for f, w in sorted(zip(FEATS, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {w:.3f}")

joblib.dump(model, OUT / "eta_xgb.joblib")
print("Saved:", OUT / "eta_xgb.joblib")