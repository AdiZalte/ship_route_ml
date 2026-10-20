
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

RAW = Path("data/interim/ais_clean.parquet")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

MAX_VESSELS = 1000
MIN_RAW_POINTS = 1000
INPUT_STEPS = 20
OUTPUT_STEPS = 5
STRIDE = 5
INTERVAL_MINUTES = 5
MAX_GAP_MINUTES = 30
MAX_JUMP_SPEED_KNOTS = 70
SEED = 42

if not RAW.exists():
    raise FileNotFoundError(f"Missing cleaned data: {RAW}")

con = duckdb.connect()
source = str(RAW.resolve()).replace("\\", "/")

con.execute(f"""
CREATE TEMP TABLE selected_vessels AS
SELECT mmsi
FROM (
    SELECT mmsi, COUNT(*) AS n
    FROM read_parquet('{source}')
    WHERE mmsi BETWEEN 100000000 AND 999999999
    GROUP BY mmsi
    HAVING COUNT(*) >= {MIN_RAW_POINTS}
)
ORDER BY  hash(MMSI)
LIMIT {MAX_VESSELS}
""")

vessels = [
    row[0]
    for row in con.execute(
        "SELECT mmsi FROM selected_vessels"
    ).fetchall()
]

if len(vessels) < 3:
    con.close()
    raise RuntimeError(
        "Too few eligible vessels. Check ais_clean.parquet."
    )

print("Selected vessels:", len(vessels))
print("Loading and sampling five-minute observations...")

df = con.execute(f"""
SELECT mmsi, timestamp, latitude, longitude, sog, cog
FROM (
    SELECT
        a.mmsi,
        a.timestamp,
        a.latitude,
        a.longitude,
        a.sog,
        a.cog,
        ROW_NUMBER() OVER (
            PARTITION BY
                a.mmsi,
                time_bucket(
                    INTERVAL '{INTERVAL_MINUTES} minutes',
                    a.timestamp
                )
            ORDER BY a.timestamp
        ) AS rn
    FROM read_parquet('{source}') a
    INNER JOIN selected_vessels s USING (mmsi)
    WHERE a.latitude BETWEEN -90 AND 90
      AND a.longitude BETWEEN -180 AND 180
      AND a.sog BETWEEN 0 AND 60
      AND a.cog >= 0 AND a.cog < 360
      AND a.timestamp IS NOT NULL
)
WHERE rn = 1
ORDER BY mmsi, timestamp
""").df()

con.close()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.drop_duplicates(["mmsi", "timestamp"])
df = df.sort_values(["mmsi", "timestamp"])

rng = np.random.default_rng(SEED)
vessels = np.array(vessels)
rng.shuffle(vessels)

n = len(vessels)
train_ids = set(vessels[:int(n * 0.70)])
val_ids = set(vessels[int(n * 0.70):int(n * 0.85)])
test_ids = set(vessels[int(n * 0.85):])

splits = {
    "train": ([], []),
    "val": ([], []),
    "test": ([], [])
}

report = {
    "selected_vessels": n,
    "segments": 0,
    "discarded_short_segments": 0,
    "input_steps": INPUT_STEPS,
    "output_steps": OUTPUT_STEPS,
    "stride": STRIDE,
    "interval_minutes": INTERVAL_MINUTES,
    "max_gap_minutes": MAX_GAP_MINUTES,
    "max_jump_speed_knots": MAX_JUMP_SPEED_KNOTS,
    "feature_names": [
        "latitude", "longitude", "sog",
        "sin_cog", "cos_cog"
    ],
    "target_names": ["latitude", "longitude"]
}


def split_name(mmsi):
    if mmsi in train_ids:
        return "train"
    if mmsi in val_ids:
        return "val"
    return "test"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def make_windows(segment, name):
    values = segment[
        ["latitude", "longitude", "sog", "cog"]
    ].to_numpy(dtype=np.float32)

    features = np.column_stack([
        values[:, 0],
        values[:, 1],
        values[:, 2],
        np.sin(np.radians(values[:, 3])),
        np.cos(np.radians(values[:, 3]))
    ]).astype(np.float32)

    coords = values[:, :2]

    total = INPUT_STEPS + OUTPUT_STEPS
    if len(segment) < total:
        report["discarded_short_segments"] += 1
        return

    xs, ys = splits[name]

    for start in range(0, len(segment) - total + 1, STRIDE):
        middle = start + INPUT_STEPS
        end = middle + OUTPUT_STEPS

        xs.append(features[start:middle])
        ys.append(coords[middle:end])


for mmsi, track in df.groupby("mmsi", sort=False):
    track = track.sort_values("timestamp").reset_index(drop=True)

    if len(track) < INPUT_STEPS + OUTPUT_STEPS:
        continue

    times = track["timestamp"].astype("int64").to_numpy()
    dt = np.diff(times) / 1e9

    lat = track["latitude"].to_numpy()
    lon = track["longitude"].to_numpy()

    distance_km = haversine_km(
        lat[:-1], lon[:-1], lat[1:], lon[1:]
    )

    hours = dt / 3600
    speed_knots = np.divide(
        distance_km * 0.539957,
        hours,
        out=np.full_like(distance_km, np.inf),
        where=hours > 0
    )

    breaks = np.where(
        (dt <= 0)
        | (dt > MAX_GAP_MINUTES * 60)
        | (speed_knots > MAX_JUMP_SPEED_KNOTS)
    )[0] + 1

    boundaries = np.concatenate([
        [0], breaks, [len(track)]
    ])

    name = split_name(int(mmsi))

    for i in range(len(boundaries) - 1):
        segment = track.iloc[
            boundaries[i]:boundaries[i + 1]
        ]

        if len(segment) >= INPUT_STEPS + OUTPUT_STEPS:
            report["segments"] += 1
            make_windows(segment, name)
        else:
            report["discarded_short_segments"] += 1

    if report["segments"] and report["segments"] % 1000 == 0:
        print("Segments processed:", report["segments"])


def stack_windows(items, shape):
    if not items:
        return np.empty((0, *shape), dtype=np.float32)
    return np.stack(items).astype(np.float32)


arrays = {}

for name, (xs, ys) in splits.items():
    arrays[f"X_{name}"] = stack_windows(
        xs, (INPUT_STEPS, 5)
    )
    arrays[f"y_{name}"] = stack_windows(
        ys, (OUTPUT_STEPS, 2)
    )

    report[f"{name}_sequences"] = len(xs)
    print(
        name,
        "X:", arrays[f"X_{name}"].shape,
        "y:", arrays[f"y_{name}"].shape
    )

if not len(arrays["X_train"]):
    raise RuntimeError(
        "No training sequences were generated. "
        "Inspect the data and relax filters if appropriate."
    )

np.savez_compressed(
    OUT / "trajectory_sequences.npz",
    **arrays
)

with open(OUT / "sequence_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Saved:", OUT / "trajectory_sequences.npz")
print("Report:", OUT / "sequence_report.json")
