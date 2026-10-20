"""
inference.py
------------
Single entry point for using the trained models from a backend.

    from src.inference import ShipRoutePredictor
    p = ShipRoutePredictor("models")
    result = p.predict(track, wave_height=1.2, wind_speed=20.0)

`track` is a list of dicts (or a DataFrame) with the last >= 20 AIS observations,
5 minutes apart, oldest first, with keys: latitude, longitude, sog (knots),
cog (degrees, 0-360) and optionally timestamp. Only the last 20 are used.

Smoke test on a real test window (from the repo root):
    python -m src.inference --index 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

INPUT_STEPS = 20
OUTPUT_STEPS = 5
STEP_MIN = 5
OFF_LANE_KM = 3.0
ETA_MIN_KM, ETA_MAX_KM, ETA_MIN_SOG = 5.0, 200.0, 3.0   # same filters as ETA training
MAX_ETA_MIN = 1440

PORT_NAMES = ["Los Angeles", "Long Beach", "San Francisco", "Seattle",
              "Houston", "New Orleans", "Miami", "New York"]
PORTS = np.array([
    [33.72, -118.27], [33.75, -118.20], [37.80, -122.39], [47.60, -122.35],
    [29.68, -95.02], [29.94, -90.06], [25.77, -80.17], [40.67, -74.05],
])
ETA_FEATS = ["dist_km", "sog", "sog_mean6", "sin_cog", "cos_cog", "cos_to_port"]


def _hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


class ShipRoutePredictor:
    """Loads all saved models once; `predict` is cheap and thread-safe for reads."""

    def __init__(self, models_dir: str | Path = "models"):
        d = Path(models_dir)

        # trajectory (LSTM without weather: the better model in our tests)
        if not (d / "scalers.joblib").exists():
            raise FileNotFoundError(f"{d / 'scalers.joblib'} not found. "
                                    "Run: python -m src.models.train_lstm")
        self.scalers = joblib.load(d / "scalers.joblib")
        self._lstm = self._load_lstm(d / "lstm.pt")

        # ETA (XGBoost without weather, 6 features)
        eta_path = d / "eta_xgb.joblib"
        if not eta_path.exists():
            raise FileNotFoundError(f"{eta_path} not found. Run: python -m src.models.train_eta")
        self.eta_model = joblib.load(eta_path)
        if getattr(self.eta_model, "n_features_in_", len(ETA_FEATS)) != len(ETA_FEATS):
            raise ValueError("eta_xgb.joblib was not trained with the 6 no-weather features. "
                             "Rerun: python -m src.models.train_eta_weather (it re-saves eta_xgb.joblib).")

        # lanes
        lanes_path = d / "lanes.joblib"
        if not lanes_path.exists():
            raise FileNotFoundError(f"{lanes_path} not found. Run: python -m src.models.lanes_and_map")
        lanes = joblib.load(lanes_path)
        self.lane_tree = BallTree(np.radians(lanes[["lat", "lon"]].to_numpy(float)),
                                  metric="haversine")

        # risk classifier (optional; needs weather inputs)
        self.risk_model, self.risk_threshold = None, None
        if (d / "risk_xgb.joblib").exists() and (d / "risk_threshold.json").exists():
            self.risk_model = joblib.load(d / "risk_xgb.joblib")
            with open(d / "risk_threshold.json") as f:
                self.risk_threshold = float(json.load(f)["threshold"])

    # ------------------------------------------------------------------ LSTM
    @staticmethod
    def _load_lstm(path: Path):
        """Return forward(xs: (1,20,5) float32 ndarray) -> (1,5,2) ndarray."""
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python -m src.models.train_lstm")
        import torch
        import torch.nn as nn

        class LSTMModel(nn.Module):          # identical to train_lstm.py
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
                self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1]).reshape(-1, 5, 2)

        model = LSTMModel()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        def forward(xs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return model(torch.tensor(xs, dtype=torch.float32)).numpy()

        return forward

    # ------------------------------------------------------------ validation
    @staticmethod
    def _parse_track(track):
        df = pd.DataFrame(track)
        missing = {"latitude", "longitude", "sog", "cog"} - set(df.columns)
        if missing:
            raise ValueError(f"track is missing fields: {sorted(missing)}")
        if len(df) < INPUT_STEPS:
            raise ValueError(f"need at least {INPUT_STEPS} observations "
                             f"({STEP_MIN} minutes apart), got {len(df)}")
        df = df.tail(INPUT_STEPS).reset_index(drop=True)

        vals = df[["latitude", "longitude", "sog", "cog"]].apply(pd.to_numeric, errors="coerce")
        arr = vals.to_numpy(float)
        if not np.isfinite(arr).all():
            raise ValueError("track contains missing or non-numeric values")
        lat, lon, sog, cog = arr.T
        if (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any():
            raise ValueError("latitude must be within +-90 and longitude within +-180")
        if ((sog < 0) | (sog > 60)).any():
            raise ValueError("sog must be between 0 and 60 knots")
        if ((cog < 0) | (cog >= 360)).any():
            raise ValueError("cog must be in [0, 360) degrees (AIS 'not available' = 360 is not accepted)")

        warnings = []
        if "timestamp" in df.columns and df["timestamp"].notna().all():
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if ts.isna().any():
                raise ValueError("timestamp could not be parsed")
            if not ts.is_monotonic_increasing:
                raise ValueError("track must be ordered oldest to newest")
            gaps = ts.diff().dropna().dt.total_seconds() / 60.0
            if not gaps.between(3, 8).all():
                warnings.append("observations are not ~5 minutes apart; the model expects "
                                "5-minute spacing, so the prediction may be unreliable")
        return lat, lon, sog, cog, warnings

    # --------------------------------------------------------------- predict
    def predict(self, track, wave_height: float | None = None,
                wind_speed: float | None = None) -> dict:
        lat, lon, sog, cog, warnings = self._parse_track(track)
        last_lat, last_lon = float(lat[-1]), float(lon[-1])
        mean_sog_window = float(sog.mean())
        sog_mean6 = float(sog[-6:].mean())
        sin_c, cos_c = np.sin(np.radians(cog)), np.cos(np.radians(cog))

        # 1. trajectory ------------------------------------------------------
        X = np.column_stack([lat, lon, sog, sin_c, cos_c]).astype(np.float32)   # (20, 5)
        xs = self.scalers["x"].transform(X).reshape(1, INPUT_STEPS, 5).astype(np.float32)
        p = np.asarray(self._lstm(xs)).reshape(-1, 2)
        disp = self.scalers["y"].inverse_transform(p)                           # (5, 2)
        pred = disp + np.array([last_lat, last_lon])
        trajectory = [{"minutes_ahead": (i + 1) * STEP_MIN,
                       "latitude": round(float(pred[i, 0]), 5),
                       "longitude": round(float(pred[i, 1]), 5)}
                      for i in range(OUTPUT_STEPS)]
        if mean_sog_window < 1.0:
            warnings.append("vessel is nearly stationary (mean SOG < 1 kn); the trajectory model "
                            "was trained on moving vessels, so treat this prediction as low confidence")

        # 2. ETA to nearest port ---------------------------------------------
        dist = _hav(last_lat, last_lon, PORTS[:, 0], PORTS[:, 1])
        k = int(np.argmin(dist))
        d_near = float(dist[k])
        eta = {"port": PORT_NAMES[k], "distance_km": round(d_near, 2),
               "applicable": False, "eta_minutes": None, "baseline_eta_minutes": None,
               "reason": None}
        if not (ETA_MIN_KM < d_near < ETA_MAX_KM):
            eta["reason"] = (f"nearest port is {d_near:.0f} km away; ETA is only supported "
                             f"between {ETA_MIN_KM:.0f} and {ETA_MAX_KM:.0f} km")
        elif sog_mean6 <= ETA_MIN_SOG:
            eta["reason"] = f"vessel is too slow (mean SOG of last 6 steps <= {ETA_MIN_SOG:.0f} kn)"
        else:
            brg = float(_bearing(last_lat, last_lon, PORTS[k, 0], PORTS[k, 1]))
            c_last = float(np.radians(cog[-1]))
            feats = pd.DataFrame([{
                "dist_km": d_near, "sog": float(sog[-1]), "sog_mean6": sog_mean6,
                "sin_cog": float(sin_c[-1]), "cos_cog": float(cos_c[-1]),
                "cos_to_port": float(np.cos(c_last - brg)),
            }])[ETA_FEATS]
            eta_min = float(np.clip(self.eta_model.predict(feats)[0], 0, MAX_ETA_MIN))
            baseline = float(np.clip(d_near / max(sog[-1] * 1.852, 0.5) * 60, 0, MAX_ETA_MIN))
            eta.update(applicable=True, eta_minutes=round(eta_min, 1),
                       baseline_eta_minutes=round(baseline, 1))

        # 3. lane / deviation flag -------------------------------------------
        dd, _ = self.lane_tree.query(np.radians([[last_lat, last_lon]]))
        lane_km = float(dd[0, 0] * 6371.0)
        lane = {"distance_to_lane_km": round(lane_km, 2),
                "off_lane": bool(lane_km > OFF_LANE_KM),
                "threshold_km": OFF_LANE_KM}

        # 4. risk (needs weather) --------------------------------------------
        risk = {"available": False, "probability": None, "high_risk": None,
                "threshold": self.risk_threshold,
                "reason": None}
        if self.risk_model is None:
            risk["reason"] = "risk model files not found"
        else:
            n_feat = int(self.risk_model.n_features_in_)
            if n_feat > 6 and (wave_height is None or wind_speed is None):
                risk["reason"] = "wave_height and wind_speed are required for the risk model"
            else:
                f = [last_lat, last_lon, float(sog[-1]), mean_sog_window,
                     float(sin_c[-1]), float(cos_c[-1]),
                     0.0 if wave_height is None else float(wave_height),
                     0.0 if wind_speed is None else float(wind_speed)][:n_feat]
                prob = float(self.risk_model.predict_proba(np.array([f], dtype=np.float32))[0, 1])
                risk.update(available=True, probability=round(prob, 3),
                            high_risk=bool(prob >= self.risk_threshold))

        return {
            "horizon_minutes": OUTPUT_STEPS * STEP_MIN,
            "last_position": {"latitude": round(last_lat, 5), "longitude": round(last_lon, 5)},
            "trajectory": trajectory,
            "eta": eta,
            "lane": lane,
            "risk": risk,
            "warnings": warnings,
        }


def _demo(index: int, models_dir: str) -> None:
    data = np.load("data/processed/trajectory_sequences.npz")
    X, Y = data["X_test"], data["y_test"]
    window = X[index]
    cog = np.minimum(np.degrees(np.arctan2(window[:, 3], window[:, 4])) % 360.0, 359.999)
    track = [{"latitude": float(window[i, 0]), "longitude": float(window[i, 1]),
              "sog": float(window[i, 2]), "cog": float(cog[i])} for i in range(INPUT_STEPS)]

    wave = wind = None
    if "T_test" in data.files:
        try:
            from src.data.weather_join import add_weather
            w, s = add_weather(window[-1:, 0], window[-1:, 1], data["T_test"][index:index + 1])
            wave, wind = float(w[0]), float(s[0])
        except Exception as exc:                      # weather is optional for the demo
            print("(no weather for demo:", exc, ")")

    result = ShipRoutePredictor(models_dir).predict(track, wave, wind)
    print(json.dumps(result, indent=2))
    final_pred = result["trajectory"][-1]
    err = _hav(Y[index, -1, 0], Y[index, -1, 1], final_pred["latitude"], final_pred["longitude"])
    print(f"\nActual position at +25 min: {Y[index, -1].tolist()}")
    print(f"Prediction error at +25 min: {float(err):.3f} km")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="index into X_test")
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    _demo(args.index, args.models)
