import pandas as pd
from pathlib import Path

folder = Path("data/raw/ais_noaa_2024")

for file in folder.glob("*.parquet"):
    df = pd.read_parquet(file)

    print("\nFILE:", file.name)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print(df.dtypes)