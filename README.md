# ML-Based Ship Route Prediction Using AIS and Weather Data

Given a vessel's recent AIS track, the system predicts its path for the next 25 minutes, estimates the ETA to the nearest major US port, flags positions away from busy shipping lanes, and estimates the probability of a course deviation.

Built on NOAA 2024 AIS data for US waters (January 2024 log file) with hourly wave and wind data from Open-Meteo.

## Quick start: use the trained models

The six small model files needed for predictions are committed in `models/` (about 7 MB), so you can serve predictions without any data or training.

```
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`, or call it directly:

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/sample_request.json
```

Full API contract, input rules, units and caveats: **[HANDOVER.md](HANDOVER.md)**.

From Python:
```python
from src.inference import ShipRoutePredictor
result = ShipRoutePredictor("models").predict(track, wave_height=1.2, wind_speed=20.0)
```

## Reproduce training

Download the data and place the files here (not committed, they are large):

```
data/raw/ais_noaa_2024/2024_NOAA_AIS_logs_01.parquet
data/raw/ais_noaa_2024/2024_NOAA_AIS_ships_01.parquet
```

(Kaggle: "2024 US Coast Guard + NOAA AIS Dataset".) Then run from the repo root:

```
python -m src.data.clean                  # clean raw AIS
python -m src.data.build_sequences        # 20-step input / 5-step output windows, 5-minute steps
python -m src.models.evaluate_baselines   # stationary and course-speed baselines
python -m src.models.train_lstm           # trajectory LSTM          -> ADE / FDE
python -m src.models.train_eta            # ETA XGBoost              -> MAE / RMSE / R2

# weather experiments (needs internet; fetch is cached after the first run)
python -m src.data.fetch_weather          # hourly wave height and wind per 1-degree cell
python -m src.models.train_lstm_weather   # LSTM with weather
python -m src.models.train_eta_weather    # ETA with weather (also re-saves eta_xgb.joblib)
python -m src.models.train_risk           # course-deviation risk classifier

python -m src.models.lanes_and_map        # DBSCAN lanes, off-lane flag, reports/route_map.html
python -m src.inference --index 0         # smoke test of the serving code on a test window
```

Rough runtimes on CPU: build_sequences 3-5 min, train_lstm 10-15 min, train_eta 3 min, fetch_weather 10-20 min, each weather model 5-15 min, lanes_and_map 3-5 min.

## Method

| Part | Approach |
|---|---|
| Trajectory | 2-layer LSTM (64 hidden) on 20 observations (lat, lon, SOG, sin/cos COG) at 5-minute spacing; predicts displacement for the next 5 steps. Vessels split 70/15/15 by MMSI. Windows with mean SOG under 1 knot are skipped. |
| ETA | XGBoost regressor. Destination = nearest of 8 US ports; target = minutes until within 5 km of it (ships above 3 knots, 5-200 km away, arriving within 24 h). Features: distance to port, SOG, mean SOG, course, heading toward port. |
| Lanes | DBSCAN (haversine) on traffic-weighted 0.02-degree grid cells of moving-ship positions. A position more than 3 km from any lane cell is flagged off-lane. |
| Risk | XGBoost classifier. Label = 25-minute course-speed prediction error above the training 90th percentile (4.63 km). Features: position, SOG, course, wave height, wind speed. Threshold chosen on validation to maximize F2. |

## Results (held-out vessels)

**Trajectory, 25 minutes ahead (moving ships)**

| Method | ADE (km) | FDE (km) |
|---|---|---|
| Stationary baseline | 2.576 | 4.189 |
| Course-speed baseline | 0.994 | 1.863 |
| **LSTM** | **0.915** | **1.619** |
| LSTM with weather | 0.920 | 1.626 |

**ETA (56,600 samples, 716 vessels)**

| Method | MAE (min) | RMSE (min) | R2 |
|---|---|---|---|
| Baseline distance / SOG | 224.1 | 373.7 | 0.123 |
| **XGBoost** | **180.1** | **269.0** | **0.546** |
| XGBoost with weather | 188.2 | 272.8 | 0.533 |

**Risk classifier (test deviation rate 11.8%)**

| Features | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| Without weather | 0.363 | 0.627 | 0.460 | 0.547 |
| With weather | 0.383 | 0.594 | 0.466 | 0.535 |

**Lanes**: about 255-270 clusters (the position sample is not seeded, so the count varies slightly between runs), about 17.5% noise cells, silhouette about -0.6. Off-lane share of test windows: 4.2% (> 2 km), 2.2% (> 3 km), 1.4% (> 5 km), 1.1% (> 10 km).

Weather features did not measurably improve trajectory, ETA or risk prediction, so the serving LSTM and ETA models do not use weather. Only the risk model takes wave height and wind speed.

## Limitations

- 25-minute horizon only; one month of US-waters data; plain LSTM rather than Bi-LSTM / Seq2Seq with attention.
- ETA destination is the nearest of 8 fixed ports, not the ship's declared destination.
- Lanes are high-traffic zones, not individual routes. Two clusters hold about 69% of lane traffic, which is why the silhouette score is low (lanes are long and thin).
- Weather is observed hourly reanalysis on a 1-degree grid at the current position, not a forecast. Wave data is coarse near the coast.
- The risk label is derived from prediction error, so it partly reflects ship speed.

## Possible next steps

Forecast weather and weather along the route, longer horizons, Bi-LSTM / Seq2Seq with attention, HDBSCAN, a label based on actual route deviation, and latency measurements.

## Layout

```
src/data/       clean.py, build_sequences.py, fetch_weather.py, weather_join.py, inspect_sequences.py
src/models/     train_lstm.py, train_eta.py, train_lstm_weather.py, train_eta_weather.py,
                train_risk.py, evaluate_baselines.py, lanes_and_map.py
src/inference.py  src/api.py        serving code (see HANDOVER.md)
models/         six small trained model files
examples/       sample_request.json
```