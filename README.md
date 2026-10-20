# ML-Based Ship Route Prediction Using Ocean and Weather Data

College ML project — NOAA 2024 AIS + Open-Meteo weather reanalysis.

## Run Order

```
# 1. Clean raw AIS
python -m src.data.clean                   # ~1 min

# 2. Build trajectory sequences (now includes timestamps T_*)
python -m src.data.build_sequences         # ~3-5 min

# 3. Fetch weather grid (Open-Meteo; cached after first run)
python -m src.data.fetch_weather           # ~10-20 min first run, <1 min cached

# 4. Train base LSTM (no weather)
python -m src.models.train_lstm            # ~5-10 min CPU

# 5. Train LSTM with weather
python -m src.models.train_lstm_weather    # ~5-10 min CPU

# 6. Train base ETA model
python -m src.models.train_eta             # ~2-3 min

# 7. Train ETA with weather
python -m src.models.train_eta_weather     # ~2-3 min

# 8. Train risk classifier
python -m src.models.train_risk            # ~1-2 min

# 9. Generate lane map
python -m src.models.lanes_and_map         # ~3-5 min
```

All commands run from the **repo root**.

## Approximate Total Runtime (CPU only)

~30–50 minutes on first run (dominated by weather fetch). Subsequent runs skip the API calls (disk-cached).

## Outputs

| File | Description |
|---|---|
| `data/processed/trajectory_sequences.npz` | X/y/T arrays per split |
| `data/external/weather_grid.parquet` | Hourly weather per 1-deg grid cell |
| `models/lstm.pt` + `scalers.joblib` | Base trajectory LSTM |
| `models/lstm_weather.pt` + `scalers_weather.joblib` | Weather LSTM |
| `models/eta_xgb.joblib` | ETA XGBoost (no weather) |
| `models/eta_weather_xgb.joblib` | ETA XGBoost (with weather) |
| `models/risk_xgb.joblib` | Route deviation classifier |
| `models/risk_threshold.json` | F2-optimal decision threshold |
| `models/lanes.joblib` | DBSCAN lane cells |
| `reports/route_map.html` | Interactive Folium map |
| `reports/results.md` | Result tables |

## Key Results (baseline)

- LSTM ADE: 0.915 km vs course-speed 0.994 km
- ETA XGBoost MAE: 180 min vs baseline 224 min (R² 0.546)

See `reports/results.md` for full tables.
