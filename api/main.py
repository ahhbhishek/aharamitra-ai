"""FastAPI application for the Aharamitra prediction service.

Endpoints:
    GET  /health        -> service health + model load status
    GET  /foods         -> food catalog (source of truth for UI dropdowns)
    GET  /regions       -> distinct regions
    GET  /festivals     -> distinct festivals
    POST /predict       -> glucose risk + safe portion prediction

Run:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from aharamitra import __version__
from aharamitra.config import settings
from aharamitra.features import INPUT_FEATURES
from aharamitra.foods import FOODS, festivals as _festivals, regions as _regions
from aharamitra.schemas import (
    FoodInfo,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="Aharamitra API",
    description="AI-Based Food Risk & Portion Intelligence — predict glucose-spike risk and safe portion size.",
    version=__version__,
)

# CORS: allow the Vite dev server (and any local origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

class ModelBundle:
    """Holds loaded model artifacts; loaded once on first prediction."""

    def __init__(self) -> None:
        self.classifier = None
        self.regressor = None
        self.risk_encoder = None
        self.metadata = None
        self._loaded = False

    def load(self) -> None:
        clf_path = settings.models_dir / settings.classifier_artifact
        reg_path = settings.models_dir / settings.regressor_artifact
        enc_path = settings.models_dir / "risk_encoder.joblib"
        meta_path = settings.models_dir / settings.metadata_artifact

        if not clf_path.exists():
            raise FileNotFoundError(
                f"Classifier artifact not found at {clf_path}. "
                "Run `python -m aharamitra.train` first."
            )

        self.classifier = joblib.load(clf_path)
        self.regressor = joblib.load(reg_path)
        self.risk_encoder = joblib.load(enc_path)
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded


bundle = ModelBundle()


@lru_cache(maxsize=1)
def _foods_cache() -> tuple[FoodInfo, ...]:
    return tuple(FoodInfo(**f) for f in FOODS)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check."""
    loaded = bundle.loaded
    # Try to load lazily if artifacts exist.
    if not loaded:
        try:
            bundle.load()
            loaded = True
        except Exception:
            loaded = False
    return HealthResponse(status="ok", version=__version__, models_loaded=loaded)


@app.get("/foods", response_model=list[FoodInfo])
def foods() -> list[FoodInfo]:
    """Return the full food catalog (canonical source for UI dropdowns)."""
    return list(_foods_cache())


@app.get("/regions", response_model=list[str])
def regions() -> list[str]:
    return _regions()


@app.get("/festivals", response_model=list[str])
def festivals() -> list[str]:
    return _festivals()


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    """Predict glucose-spike risk and safe portion size."""
    if not bundle.loaded:
        try:
            bundle.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Models not available: {e}")

    import pandas as pd

    sample = req.to_model_input()
    X = pd.DataFrame([sample])[INPUT_FEATURES]

    risk_num = int(bundle.classifier.predict(X)[0])
    risk_label = str(bundle.risk_encoder.inverse_transform([risk_num])[0])
    portion = float(bundle.regressor.predict(X)[0])

    try:
        proba = bundle.classifier.predict_proba(X)
        confidence = float(proba.max(axis=1)[0])
    except Exception:
        confidence = None

    return PredictionResponse(
        glucose_spike_risk=risk_label,
        risk_encoded=risk_num,
        safe_portion_count=round(portion, 2),
        confidence=round(confidence, 4) if confidence is not None else None,
        food_name=req.food_name,
        festival=req.festival,
        region=req.region,
    )


@app.get("/")
def root() -> dict:
    return {
        "name": "Aharamitra API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/foods", "/regions", "/festivals", "/predict"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
