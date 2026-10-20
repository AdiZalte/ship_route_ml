"""
api.py
------
Minimal HTTP API around ShipRoutePredictor.

Run from the repo root:
    pip install fastapi uvicorn
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs
Environment variables:
    MODELS_DIR       folder with the saved models (default: models)
    ALLOWED_ORIGINS  comma-separated CORS origins for the frontend (default: *)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import PORT_NAMES, PORTS, ShipRoutePredictor

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["predictor"] = ShipRoutePredictor(os.environ.get("MODELS_DIR", "models"))
    yield
    _state.clear()


app = FastAPI(title="Ship Route Prediction API", version="1.0", lifespan=lifespan)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["*"], allow_headers=["*"])


class Observation(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    sog: float = Field(ge=0, le=60, description="speed over ground, knots")
    cog: float = Field(ge=0, lt=360, description="course over ground, degrees")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601, optional")


class PredictRequest(BaseModel):
    track: list[Observation] = Field(
        min_length=20,
        description="At least 20 observations, 5 minutes apart, oldest first (last 20 are used)")
    wave_height: Optional[float] = Field(default=None, ge=0, description="metres, optional")
    wind_speed: Optional[float] = Field(
        default=None, ge=0,
        description="same unit as the training weather grid (see HANDOVER.md), optional")


@app.get("/health")
def health():
    return {"status": "ok", "risk_model_loaded": _state["predictor"].risk_model is not None}


@app.get("/ports")
def ports():
    return [{"name": n, "latitude": float(p[0]), "longitude": float(p[1])}
            for n, p in zip(PORT_NAMES, PORTS)]


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return _state["predictor"].predict(
            [o.model_dump() for o in req.track], req.wave_height, req.wind_speed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
