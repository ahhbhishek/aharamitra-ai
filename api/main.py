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
# Lookup tables & helpers for the human-friendly output
# ---------------------------------------------------------------------------

# {food_name -> FoodNutrition dict} for O(1) weight lookup.
_FOOD_BY_NAME = {f["food_name"]: f for f in FOODS}

# Risk label -> one-word action + emoji shown in the UI.
VERDICTS = {
    "low": "Enjoy",
    "moderate": "Go easy",
    "high": "Limit",
    "very_high": "Avoid",
}


def human_pieces(portions: float) -> str:
    """Round a fractional portion count into friendly piece language.

    Examples: 0.4 -> "½ piece", 1.0 -> "1 piece", 1.5 -> "1-2 pieces",
    2.3 -> "2-3 pieces", 3.0 -> "3 pieces".
    """
    p = max(0.0, portions)
    # Below ~0.6 we collapse to a half-piece suggestion.
    if p < 0.6:
        return "½ piece" if p > 0.1 else "Just a taste"
    low = int(round(p - 0.25))
    high = int(round(p + 0.25))
    if low < 1:
        low = 1
    if high < 1:
        high = 1
    if low == high:
        return f"{low} piece{'s' if low > 1 else ''}"
    return f"{low}-{high} pieces"


def build_reasons(req: PredictionRequest, gl: float) -> list[str]:
    """Generate 2-3 short, personalized 'why' bullets.

    Order matters: the most decisive factor (glycemic load) comes first,
    followed by the user-specific modifiers, capped at three so the card
    stays scannable.
    """
    reasons: list[str] = []

    if gl > 15:
        reasons.append(f"High glycemic load (GL {gl:.1f})")
    elif gl > 11:
        reasons.append(f"Moderate glycemic load (GL {gl:.1f})")
    else:
        reasons.append(f"Low glycemic load (GL {gl:.1f})")

    if req.sugar_per_item_g >= 12:
        reasons.append(f"{int(req.sugar_per_item_g)}g sugar per piece")
    if req.diabetes_status == 1:
        reasons.append("Diabetes raises your spike risk")
    if req.bmi_category in ("overweight", "obese"):
        reasons.append(f"Your BMI ({req.bmi:.0f}) is above the healthy range")
    if req.fasting_state == 0:
        reasons.append("On an empty stomach, the spike is sharper")
    if req.age >= 55:
        reasons.append("Slower sugar metabolism at this age")

    return reasons[:3]


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
    """Predict glucose-spike risk and safe portion size.

    The ML model still predicts a fractional ``safe_portion_count``; this
    endpoint converts it into grams, macros, a verdict, and the 'why'
    bullets before returning — so callers receive human-actionable output.
    """
    if not bundle.loaded:
        try:
            bundle.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Models not available: {e}")

    import pandas as pd

    sample = req.to_model_input()
    X = pd.DataFrame([sample])[INPUT_FEATURES]

    # ML inference (unchanged): risk class + portion count.
    risk_num = int(bundle.classifier.predict(X)[0])
    risk_label = str(bundle.risk_encoder.inverse_transform([risk_num])[0])
    portion = float(bundle.regressor.predict(X)[0])
    portion = max(0.0, min(portion, 3.0))  # clamp to the trained range

    # Resolve the food's per-portion weight (grams for ONE piece/serving).
    food = _FOOD_BY_NAME.get(req.food_name)
    if food is None:
        raise HTTPException(status_code=400, detail=f"Unknown food: {req.food_name}")
    weight_g = float(food["weight_g"])

    # Translate the abstract portion count into concrete units.
    safe_grams = max(5, round(portion * weight_g))
    safe_pieces = human_pieces(portion)
    sugar_g = round(portion * req.sugar_per_item_g, 1)
    carbs_g = round(portion * req.carbs_per_item_g, 1)
    energy_kcal = round(portion * req.energy_per_item_kcal)
    gl = round(sample["glycemic_load"], 1)

    return PredictionResponse(
        glucose_spike_risk=risk_label,
        verdict=VERDICTS.get(risk_label, "Go easy"),
        safe_grams=safe_grams,
        safe_pieces=safe_pieces,
        sugar_g=sugar_g,
        carbs_g=carbs_g,
        energy_kcal=energy_kcal,
        glycemic_load=gl,
        reasons=build_reasons(req, gl),
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
