import pandas as pd
from pathlib import Path

raw = Path("data/raw/ais_noaa_2024")

logs = pd.read_parquet(
    raw / "2024_NOAA_AIS_logs_01.parquet",
    columns=["MMSI", "BaseDateTime", "LAT", "LON"]
)

ships = pd.read_parquet(
    raw / "2024_NOAA_AIS_ships_01.parquet"
)

print("Log MMSI range:", logs["MMSI"].min(), logs["MMSI"].max())
print("Log MMSI zero:", (logs["MMSI"] == 0).sum())
print("Ship MMSI zero:", (ships["MMSI"] == 0).sum())
print("Duplicate metadata MMSI:", ships["MMSI"].duplicated().sum())

print("\nLog time range:")
print(logs["BaseDateTime"].min(), logs["BaseDateTime"].max())

print("\nPosition records per vessel:")
print(logs.groupby("MMSI").size().describe())

print("\nMetadata missing values:")
print(ships.isna().sum())