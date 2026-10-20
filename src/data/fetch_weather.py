"""
fetch_weather.py
----------------
Fetches hourly wave_height (m) and wind_speed_10m (m/s) for every 1-degree
grid cell that contains AIS positions.

Usage:
    python -m src.data.fetch_weather
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

RAW = Path("data/interim/ais_clean.parquet")
OUT_DIR = Path("data/external")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "weather_cache"
CACHE_DIR.mkdir(exist_ok=True)

OUT_PARQUET = OUT_DIR / "weather_grid.parquet"
MAX_CELLS = 1500
BATCH_SIZE = 50          # Open-Meteo accepts up to 50 locations per request
SLEEP_BETWEEN_BATCHES = 1.0  # seconds
MAX_FILL_DEG = 2.0       # max haversine distance for NaN fill

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------- helpers ----------

def _retry_get(url: str, params: dict, max_tries: int = 5) -> requests.Response:
    """GET with exponential back-off on 429 / 5xx."""
    delay = 5.0
    for attempt in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            if attempt == max_tries - 1:
                raise
            print(f"  Network error ({exc}), retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == max_tries - 1:
                r.raise_for_status()
            wait = delay
            # honour Retry-After if present
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = float(ra)
                except ValueError:
                    pass
            print(f"  HTTP {r.status_code}, retrying in {wait:.0f}s ...")
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue

        r.raise_for_status()
        return r

    raise RuntimeError("Exceeded max retries")


def _cache_key(api: str, lats: list[float], lons: list[float],
               start: str, end: str) -> str:
    raw = f"{api}_{'_'.join(f'{la}_{lo}' for la, lo in zip(lats, lons))}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()


def _fetch_marine_batch(lats: list[float], lons: list[float],
                        start: str, end: str) -> list[dict]:
    """Returns list of {lat, lon, times[], wave_height[]}."""
    cache_path = CACHE_DIR / (_cache_key("marine", lats, lons, start, end) + ".json")
    if cache_path.exists():
        with cache_path.open() as f:
            raw = json.load(f)
    else:
        params = {
            "latitude": ",".join(str(la) for la in lats),
            "longitude": ",".join(str(lo) for lo in lons),
            "hourly": "wave_height",
            "start_date": start,
            "end_date": end,
            "format": "json",
        }
        r = _retry_get(MARINE_URL, params)
        raw = r.json()
        if isinstance(raw, dict):      # single-location: wrap in list
            raw = [raw]
        with cache_path.open("w") as f:
            json.dump(raw, f)

    results = []
    for loc_i, loc in enumerate(raw):
        t_vals = loc["hourly"]["time"]
        wh = loc["hourly"].get("wave_height", [None] * len(t_vals))
        results.append({
            "req_lat": lats[loc_i], "req_lon": lons[loc_i],
            "times": t_vals, "wave_height": wh,
        })
    return results


def _fetch_archive_batch(lats: list[float], lons: list[float],
                         start: str, end: str) -> list[dict]:
    """Returns list of {lat, lon, times[], wind_speed_10m[]}."""
    cache_path = CACHE_DIR / (_cache_key("archive", lats, lons, start, end) + ".json")
    if cache_path.exists():
        with cache_path.open() as f:
            raw = json.load(f)
    else:
        params = {
            "latitude": ",".join(str(la) for la in lats),
            "longitude": ",".join(str(lo) for lo in lons),
            "hourly": "wind_speed_10m",
            "start_date": start,
            "end_date": end,
            "windspeed_unit": "ms",
            "format": "json",
        }
        r = _retry_get(ARCHIVE_URL, params)
        raw = r.json()
        if isinstance(raw, dict):
            raw = [raw]
        with cache_path.open("w") as f:
            json.dump(raw, f)

    results = []
    for loc_i, loc in enumerate(raw):
        t_vals = loc["hourly"]["time"]
        ws = loc["hourly"].get("wind_speed_10m", [None] * len(t_vals))
        results.append({
            "req_lat": lats[loc_i], "req_lon": lons[loc_i],
            "times": t_vals, "wind_speed_10m": ws,
        })
    return results


# ---------- Step 1: grid cells from AIS ----------

print("Querying AIS for date range and grid cells ...")
con = duckdb.connect()
src = str(RAW.resolve()).replace("\\", "/")

meta = con.execute(f"""
SELECT
    MIN(epoch(timestamp)) AS t_min,
    MAX(epoch(timestamp)) AS t_max
FROM read_parquet('{src}')
WHERE timestamp IS NOT NULL
""").fetchone()
t_min_epoch, t_max_epoch = meta
con.close()

start_date = datetime.fromtimestamp(t_min_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
end_date = datetime.fromtimestamp(t_max_epoch, tz=timezone.utc).strftime("%Y-%m-%d")
print(f"  AIS date range: {start_date} -> {end_date}")

con = duckdb.connect()
cells_df = con.execute(f"""
SELECT
    ROUND(latitude)  AS cell_lat,
    ROUND(longitude) AS cell_lon,
    COUNT(*) AS n
FROM read_parquet('{src}')
WHERE latitude BETWEEN -90 AND 90
  AND longitude BETWEEN -180 AND 180
  AND mmsi BETWEEN 100000000 AND 999999999
GROUP BY cell_lat, cell_lon
ORDER BY n DESC
LIMIT {MAX_CELLS}
""").df()
con.close()

cells_df = cells_df.astype({"cell_lat": float, "cell_lon": float, "n": int})
print(f"  Grid cells: {len(cells_df)} (capped at {MAX_CELLS})")


# ---------- Step 2: fetch in batches ----------

def _batch(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


lats = cells_df["cell_lat"].tolist()
lons = cells_df["cell_lon"].tolist()
pairs = list(zip(lats, lons))
n_batches = math.ceil(len(pairs) / BATCH_SIZE)

marine_rows: list[dict] = []
archive_rows: list[dict] = []

for bi, batch_pairs in enumerate(_batch(pairs, BATCH_SIZE)):
    bla = [p[0] for p in batch_pairs]
    blo = [p[1] for p in batch_pairs]
    print(f"  Marine batch {bi+1}/{n_batches} ({len(batch_pairs)} cells) ...", end=" ", flush=True)
    try:
        results = _fetch_marine_batch(bla, blo, start_date, end_date)
        marine_rows.extend(results)
        print("ok")
    except Exception as exc:
        print(f"ERROR: {exc}")
        # fill with NaN placeholders so we still have a row
        for la, lo in batch_pairs:
            marine_rows.append({"req_lat": la, "req_lon": lo, "times": [], "wave_height": []})
    if bi < n_batches - 1:
        time.sleep(SLEEP_BETWEEN_BATCHES)

for bi, batch_pairs in enumerate(_batch(pairs, BATCH_SIZE)):
    bla = [p[0] for p in batch_pairs]
    blo = [p[1] for p in batch_pairs]
    print(f"  Archive batch {bi+1}/{n_batches} ({len(batch_pairs)} cells) ...", end=" ", flush=True)
    try:
        results = _fetch_archive_batch(bla, blo, start_date, end_date)
        archive_rows.extend(results)
        print("ok")
    except Exception as exc:
        print(f"ERROR: {exc}")
        for la, lo in batch_pairs:
            archive_rows.append({"req_lat": la, "req_lon": lo, "times": [], "wind_speed_10m": []})
    if bi < n_batches - 1:
        time.sleep(SLEEP_BETWEEN_BATCHES)


# ---------- Step 3: flatten to long DataFrames ----------

def _flatten_marine(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        for t_str, wh in zip(r["times"], r["wave_height"]):
            records.append({
                "cell_lat": r["req_lat"],
                "cell_lon": r["req_lon"],
                "hour": pd.Timestamp(t_str, tz="UTC"),
                "wave_height": wh,
            })
    return pd.DataFrame(records)


def _flatten_archive(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        for t_str, ws in zip(r["times"], r["wind_speed_10m"]):
            records.append({
                "cell_lat": r["req_lat"],
                "cell_lon": r["req_lon"],
                "hour": pd.Timestamp(t_str, tz="UTC"),
                "wind_speed": ws,
            })
    return pd.DataFrame(records)


print("Assembling DataFrames ...")
df_marine = _flatten_marine(marine_rows)
df_archive = _flatten_archive(archive_rows)

grid = df_marine.merge(df_archive, on=["cell_lat", "cell_lon", "hour"], how="outer")
grid["wave_height"] = pd.to_numeric(grid["wave_height"], errors="coerce")
grid["wind_speed"] = pd.to_numeric(grid["wind_speed"], errors="coerce")

print(f"  Raw rows: {len(grid):,}")
print(f"  wave_height NaN: {grid['wave_height'].isna().sum():,} "
      f"({grid['wave_height'].isna().mean():.1%})")
print(f"  wind_speed  NaN: {grid['wind_speed'].isna().sum():,} "
      f"({grid['wind_speed'].isna().mean():.1%})")


# ---------- Step 4: fill coastal NaN in wave_height via nearest non-NaN cell ----------

print("Filling coastal NaN wave_heights with nearest good cell (<=2 deg) ...")

# Work cell-by-cell: for each (cell_lat, cell_lon) that has all-NaN wave_height,
# find the nearest cell (by haversine) that has any data, and borrow its values
# aligned by hour.

cell_wh = grid.groupby(["cell_lat", "cell_lon"])["wave_height"].mean().reset_index()
bad_cells = cell_wh[cell_wh["wave_height"].isna()][["cell_lat", "cell_lon"]].values
good_cells = cell_wh[~cell_wh["wave_height"].isna()][["cell_lat", "cell_lon"]].values

if len(bad_cells) > 0 and len(good_cells) > 0:
    tree_good = BallTree(np.radians(good_cells), metric="haversine")
    dists, idxs = tree_good.query(np.radians(bad_cells))
    dists_deg = np.degrees(dists[:, 0])

    n_filled = 0
    for bi, (bad_la, bad_lo) in enumerate(bad_cells):
        if dists_deg[bi] > MAX_FILL_DEG:
            continue  # too far: leave NaN
        src_la, src_lo = good_cells[idxs[bi, 0]]
        src_wh = grid[(grid["cell_lat"] == src_la) & (grid["cell_lon"] == src_lo)][
            ["hour", "wave_height"]
        ].rename(columns={"wave_height": "wh_fill"})

        mask = (grid["cell_lat"] == bad_la) & (grid["cell_lon"] == bad_lo) & grid["wave_height"].isna()
        grid = grid.merge(src_wh, on="hour", how="left")
        grid.loc[mask, "wave_height"] = grid.loc[mask, "wh_fill"]
        grid = grid.drop(columns=["wh_fill"])
        n_filled += int(mask.sum())

    print(f"  Filled {n_filled:,} wave_height NaN cells from nearest neighbour")
else:
    print("  No coastal NaN cells to fill, or no good reference cells")

# ---------- Step 5: save ----------

grid = grid.reset_index(drop=True)
print(f"Final grid rows: {len(grid):,}")
nan_check = grid[["wave_height", "wind_speed"]].isna().mean()
print("  Residual NaN fractions:", nan_check.to_dict())

grid.to_parquet(OUT_PARQUET, index=False)
print(f"Saved: {OUT_PARQUET}")
