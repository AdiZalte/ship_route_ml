
from pathlib import Path
import duckdb

raw = Path("data/raw/ais_noaa_2024/2024_NOAA_AIS_logs_01.parquet")
out = Path("data/interim/ais_clean.parquet")
out.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
source = str(raw.resolve()).replace("\\", "/")
target = str(out.resolve()).replace("\\", "/")

con.execute(f"""
COPY (
    SELECT
        MMSI AS mmsi,
        BaseDateTime AS timestamp,
        LAT AS latitude,
        LON AS longitude,
        SOG AS sog,
        COG AS cog,
        Heading AS heading
    FROM read_parquet('{source}')
    WHERE MMSI > 0
      AND LAT BETWEEN -90 AND 90
      AND LON BETWEEN -180 AND 180
      AND SOG BETWEEN 0 AND 60
      AND COG >= 0 AND COG < 360
      AND BaseDateTime IS NOT NULL
    ORDER BY MMSI, BaseDateTime
) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("Cleaned data saved:", out)
con.close()
