import pandas as pd
from pathlib import Path

raw = Path("data/raw/ais_noaa_2024")
out = Path("data/interim")
out.mkdir(parents=True, exist_ok=True)

logs = raw / "2024_NOAA_AIS_logs_01.parquet"
ships = raw / "2024_NOAA_AIS_ships_01.parquet"

# Read only required columns and sample rows.
df = pd.read_parquet(
    logs,
    columns=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading"]
)

df = df.sample(n=min(1_000_000, len(df)), random_state=42)

df = df.rename(columns={
    "MMSI": "mmsi",
    "BaseDateTime": "timestamp",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading"
})

df = df.dropna(subset=[
    "mmsi", "timestamp", "latitude",
    "longitude", "sog", "cog"
])

df = df[
    df["latitude"].between(-90, 90)
    & df["longitude"].between(-180, 180)
    & df["sog"].between(0, 60)
    & df["cog"].between(0, 360)
]

df = df.sort_values(["mmsi", "timestamp"])
df.to_parquet(out / "ais_sample.parquet", index=False)

print("Sample rows:", len(df))
print("Unique MMSI:", df["mmsi"].nunique())
print("Saved to:", out / "ais_sample.parquet")