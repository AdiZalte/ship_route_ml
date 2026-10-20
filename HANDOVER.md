# Ship Route Prediction: handover for backend and frontend

Given the last 20 AIS observations of a vessel (5 minutes apart), the system returns:

1. **Trajectory**: predicted positions for the next 25 minutes (5 points, 5 min apart).
2. **ETA**: minutes to the nearest of 8 US ports, when the vessel is heading in (see limits below).
3. **Lane / deviation flag**: distance to the nearest busy shipping lane and an off-lane flag.
4. **Risk**: probability of a course deviation (optional, needs wave height and wind speed).

## Files the backend needs

Copy these into the deployment (all produced by the training scripts):

| File | Produced by |
|---|---|
| `models/lstm.pt`, `models/scalers.joblib` | `python -m src.models.train_lstm` |
| `models/eta_xgb.joblib` | `python -m src.models.train_eta` (also re-saved by `train_eta_weather`) |
| `models/lanes.joblib` | `python -m src.models.lanes_and_map` |
| `models/risk_xgb.joblib`, `models/risk_threshold.json` | `python -m src.models.train_risk` (optional) |

Code: `src/inference.py` (the predictor) and `src/api.py` (HTTP wrapper), plus an empty `src/__init__.py`.

Python packages: `torch` (CPU is enough), `xgboost`, `scikit-learn`, `joblib`, `numpy`, `pandas`, `fastapi`, `uvicorn`.

## Run

From the repo root:

```
pip install fastapi uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Environment variables: `MODELS_DIR` (default `models`) and `ALLOWED_ORIGINS` (comma-separated CORS origins for the frontend, default `*`, restrict it in production).

Models load once at startup. Each prediction is a few milliseconds on CPU.

Python use without HTTP:
```python
from src.inference import ShipRoutePredictor
result = ShipRoutePredictor("models").predict(track, wave_height=1.2, wind_speed=20.0)
```

Smoke test on a real held-out window (needs the processed data): `python -m src.inference --index 0`. It prints the full response and the real 25-minute error in km.

## API

### `GET /health`
`{"status": "ok", "risk_model_loaded": true}`

### `GET /ports`
The 8 supported ports: `[{"name": "Los Angeles", "latitude": 33.72, "longitude": -118.27}, ...]`

### `POST /predict`

Request:
```json
{
  "track": [
    {"latitude": 33.80, "longitude": -118.30, "sog": 12.0, "cog": 120.0, "timestamp": "2024-01-15T10:00:00Z"}
  ],
  "wave_height": 1.5,
  "wind_speed": 25.0
}
```

- `track`: at least 20 observations, oldest first, 5 minutes apart. Only the last 20 are used. `timestamp` is optional; if sent, the order is checked and a warning is returned when spacing is not about 5 minutes.
- `latitude` in [-90, 90], `longitude` in [-180, 180], `sog` in knots [0, 60], `cog` in degrees [0, 360). AIS "course not available" (360) must be filtered out before calling.
- `wave_height` (metres) and `wind_speed` are optional. Without both, `risk.available` is false and everything else still works.

Response (values are illustrative):
```json
{
  "horizon_minutes": 25,
  "last_position": {"latitude": 33.819, "longitude": -118.281},
  "trajectory": [
    {"minutes_ahead": 5,  "latitude": 33.82195, "longitude": -118.27799},
    {"minutes_ahead": 10, "latitude": 33.82495, "longitude": -118.27488}
  ],
  "eta": {
    "port": "Long Beach", "distance_km": 10.72, "applicable": true,
    "eta_minutes": 41.0, "baseline_eta_minutes": 28.9, "reason": null
  },
  "lane": {"distance_to_lane_km": 0.14, "off_lane": false, "threshold_km": 3.0},
  "risk": {"available": true, "probability": 0.41, "high_risk": false, "threshold": 0.5, "reason": null},
  "warnings": []
}
```
(`trajectory` always has 5 entries, at 5, 10, 15, 20 and 25 minutes.)

Errors: `422` with a `detail` message for invalid input (too few points, NaN, out-of-range values, unordered timestamps).

## Rules the clients must know

- **ETA** is only returned (`applicable: true`) when the nearest port is between 5 and 200 km away and the mean speed of the last 6 observations is above 3 knots. Otherwise `eta_minutes` is `null` and `reason` says why. Show "ETA not available" in the UI, not 0.
- **Ports** are a fixed list of 8 (Los Angeles, Long Beach, San Francisco, Seattle, Houston, New Orleans, Miami, New York). The destination is always the nearest of these, not the ship's declared destination.
- **Trajectory** is trained on moving vessels (mean SOG at least 1 knot). For near-stationary vessels a warning is returned and the prediction is low confidence.
- **Lane / off-lane**: lanes are high-traffic zones learned from January 2024 US AIS data. `off_lane` means the last position is more than 3 km from any lane cell. It flags a possible deviation; it does not say why.
- **Risk** needs weather inputs. The model was trained with hourly wave height (metres) and 10 m wind speed from Open-Meteo reanalysis at the vessel's position. **Check the wind unit before wiring a live weather source:** the training grid was fetched with a parameter the API ignores, so wind is most likely in **km/h**, not m/s. Verify with `python -c "import pandas as pd; print(pd.read_parquet('data/external/weather_grid.parquet')['wind_speed'].describe())"`. If the maximum is above about 45, it is km/h, and the live source must be converted to km/h too.
- **Coverage**: trained on one month (January 2024) of US-waters AIS. Do not expect good results elsewhere without retraining.

## Measured accuracy (held-out vessels)

| Task | Result |
|---|---|
| Trajectory, 25 min ahead (ADE / FDE) | LSTM 0.915 / 1.619 km, course-speed baseline 0.994 / 1.863 km, stationary 2.576 / 4.189 km |
| ETA | XGBoost MAE 180.1 min, RMSE 269.0, R2 0.546; baseline distance / SOG MAE 224.1, RMSE 373.7, R2 0.123 |
| Risk (course-deviation label, test deviation rate 11.8%) | precision 0.38, recall 0.59, F1 0.47 (with weather); weather did not improve it measurably |
| Lanes | 254-268 clusters, about 17.5% noise cells; off-lane in about 2% of test windows at 3 km |

Weather features were tested for trajectory, ETA and risk and did not improve results, so the production LSTM and ETA models do not use weather. Only the risk model takes weather inputs.

## Frontend notes

- Draw `last_position` plus the 5 `trajectory` points as the predicted path. The past track is whatever the client already has.
- Show `eta` as a label or popup when `applicable` is true, with `baseline_eta_minutes` as an optional comparison.
- Show `lane.off_lane` as a badge, and `risk.probability` as a gauge when `risk.available` is true (highlight when `high_risk`).
- Show `warnings` as a small banner.
- `reports/route_map.html` (folium) is a demo of the same outputs for test data, not part of the API.

## Known limitations

25-minute horizon only, one month of data, plain LSTM, 8 fixed ports, silhouette of the lane clustering is low because lanes are long and thin, weather is observed (not forecast).
