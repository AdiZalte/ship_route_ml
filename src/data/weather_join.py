"""
weather_join.py
---------------
Helper module to look up weather (wave_height, wind_speed) for AIS positions.

Usage:
    from src.data.weather_join import add_weather
    wave, wind = add_weather(lat_array, lon_array, epoch_seconds_array)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

WEATHER_PATH = Path("data/external/weather_grid.parquet")

_log = logging.getLogger(__name__)

# Module-level cache so we only load once per Python session
_grid: pd.DataFrame | None = None
_ball_tree: BallTree | None = None
_unique_cells: np.ndarray | None = None    # (M, 2)  lat/lon of unique cells
_train_medians: dict[str, float] = {}


def _load() -> None:
    global _grid, _ball_tree, _unique_cells
    if _grid is not None:
        return

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Weather grid not found at {WEATHER_PATH}. "
            "Run: python -m src.data.fetch_weather"
        )

    _grid = pd.read_parquet(WEATHER_PATH)
    _grid["hour_epoch"] = _grid["hour"].astype("int64") // 1_000_000_000  # UTC epoch seconds

    # Build BallTree over unique grid cells for fast nearest-cell lookup
    uc = _grid[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    _unique_cells = uc.values.astype(float)
    _ball_tree = BallTree(np.radians(_unique_cells), metric="haversine")


def set_train_medians(wave_median: float, wind_median: float) -> None:
    """Store training-set medians for NaN fill. Call after fitting on train data."""
    _train_medians["wave_height"] = float(wave_median)
    _train_medians["wind_speed"] = float(wind_median)


def add_weather(
    lat: np.ndarray,
    lon: np.ndarray,
    epoch_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Look up (wave_height, wind_speed) for each (lat, lon, time).

    Parameters
    ----------
    lat, lon      : 1-D float arrays of length N
    epoch_seconds : 1-D int/float array, UTC seconds since Unix epoch

    Returns
    -------
    wave_height, wind_speed : float32 arrays of length N (NaN where unavailable)
    """
    _load()

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    epoch_seconds = np.asarray(epoch_seconds, dtype=np.int64)

    n = len(lat)
    wave_out = np.full(n, np.nan, dtype=np.float32)
    wind_out = np.full(n, np.nan, dtype=np.float32)

    # Floor epoch to nearest hour
    hour_epoch = (epoch_seconds // 3600) * 3600

    # Nearest 1-deg grid cell (haversine BallTree)
    query_pts = np.radians(np.column_stack([lat, lon]))
    _, cell_idx = _ball_tree.query(query_pts, k=1)
    cell_idx = cell_idx[:, 0]

    cell_lats = _unique_cells[cell_idx, 0]
    cell_lons = _unique_cells[cell_idx, 1]

    # Group by (cell, hour) to minimise DataFrame lookups
    df = pd.DataFrame({
        "i": np.arange(n),
        "cell_lat": cell_lats,
        "cell_lon": cell_lons,
        "hour_epoch": hour_epoch,
    })

    # Merge with weather grid
    grid_lookup = _grid[["cell_lat", "cell_lon", "hour_epoch", "wave_height", "wind_speed"]].copy()
    merged = df.merge(grid_lookup, on=["cell_lat", "cell_lon", "hour_epoch"], how="left")
    # If duplicates (shouldn't happen), take first
    merged = merged.groupby("i", sort=False).first().reset_index().sort_values("i")

    wave_out = merged["wave_height"].to_numpy(dtype=np.float32)
    wind_out = merged["wind_speed"].to_numpy(dtype=np.float32)

    # Fill remaining NaN with training medians (if set)
    nan_wave = np.isnan(wave_out).sum()
    nan_wind = np.isnan(wind_out).sum()

    if nan_wave > 0 and "wave_height" in _train_medians:
        wave_out = np.where(np.isnan(wave_out), _train_medians["wave_height"], wave_out)
    if nan_wind > 0 and "wind_speed" in _train_medians:
        wind_out = np.where(np.isnan(wind_out), _train_medians["wind_speed"], wind_out)

    total_filled = nan_wave + nan_wind
    if total_filled > 0:
        _log.info(
            "Filled %d NaN weather values with training medians "
            "(wave: %d, wind: %d)",
            total_filled, nan_wave, nan_wind,
        )

    return wave_out.astype(np.float32), wind_out.astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test
    import sys
    logging.basicConfig(level=logging.INFO)
    lat_test = np.array([37.8, 33.7])
    lon_test = np.array([-122.4, -118.3])
    ts_test = np.array([1704067200, 1704070800])  # 2024-01-01 00:00 / 01:00 UTC
    wh, ws = add_weather(lat_test, lon_test, ts_test)
    print("wave_height:", wh)
    print("wind_speed: ", ws)
