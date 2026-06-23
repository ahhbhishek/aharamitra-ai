"""Pydantic request/response schemas for the Aharamitra API.

These define the contract between the React UI and the FastAPI backend,
and enforce input validation at the boundary.

Design note on inputs
---------------------
The UI asks for **natural measurements** (gender, age, height, weight) rather
than a raw BMI number. BMI and its clinical category are derived server-side
in ``PredictionRequest._derive_bmi``, so the *model* still sees exactly the
features it was trained on (``bmi``, ``bmi_category``) while the *user* never
has to know their BMI.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# Allowed categorical values (kept in sync with the food knowledge base).
BMI_CATEGORIES = ("underweight", "normal", "overweight", "obese")
GENDERS = ("male", "female")


def bmi_category(bmi: float) -> str:
    """Map a BMI value to its clinical category (WHO thresholds)."""
    if bmi < 18.5:
        return "underweight"
    if bmi < 25.0:
        return "normal"
    if bmi < 30.0:
        return "overweight"
    return "obese"


class PredictionRequest(BaseModel):
    """Inbound prediction request.

    Natural inputs (gender/height/weight) are turned into ``bmi`` and
    ``bmi_category`` by :meth:`_derive_bmi`; downstream code (and the model)
    sees those derived fields exactly as before.
    """

    # --- Natural profile inputs (what the user actually knows) ---------------
    gender: str = Field(..., description="'male' or 'female'")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    height_cm: float = Field(..., ge=100.0, le=220.0, description="Height in cm")
    weight_kg: float = Field(..., ge=30.0, le=200.0, description="Weight in kg")

    diabetes_status: int = Field(..., ge=0, le=1, description="1 = diabetic, 0 = non-diabetic")
    fasting_state: int = Field(..., ge=0, le=1, description="1 = fasting, 0 = not fasting")

    # --- Food + context ------------------------------------------------------
    festival: str
    region: str
    food_name: str
    glycemic_index: float = Field(..., ge=0, le=100)
    carbs_per_item_g: float = Field(..., ge=0, le=200)
    sugar_per_item_g: float = Field(..., ge=0, le=200)
    protein_per_item_g: float = Field(..., ge=0, le=200)
    fat_per_item_g: float = Field(..., ge=0, le=200)
    fiber_per_item_g: float = Field(..., ge=0, le=200)
    energy_per_item_kcal: float = Field(..., ge=0, le=2000)

    # Derived server-side; kept on the model so the ML feature dict is complete.
    bmi: float = Field(default=0.0, ge=10.0, le=60.0)
    bmi_category: str = Field(default="normal")

    @field_validator("gender")
    @classmethod
    def _valid_gender(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in GENDERS:
            raise ValueError(f"gender must be one of {GENDERS}")
        return v

    @model_validator(mode="after")
    def _derive_bmi(self) -> "PredictionRequest":
        """Compute BMI + category from height/weight so callers don't have to."""
        height_m = self.height_cm / 100.0
        self.bmi = round(self.weight_kg / (height_m * height_m), 1)
        self.bmi_category = bmi_category(self.bmi)
        return self

    def to_model_input(self) -> dict:
        """Convert to a dict suitable for the model, computing glycemic_load."""
        from aharamitra.labeling import glycemic_load

        d = self.model_dump()
        d["glycemic_load"] = round(glycemic_load(self.glycemic_index, self.carbs_per_item_g), 2)
        return d


class PredictionResponse(BaseModel):
    """Outbound prediction result — written for the user, not the model.

    Everything here is in plain units a non-technical person can act on:
    grams to eat, sugar/carbs/calories that represents, a one-word verdict,
    and 2-3 reasons. ML internals (encoded risk, confidence, raw portion
    count) are intentionally excluded from the public contract.
    """

    glucose_spike_risk: str                 # low | moderate | high | very_high
    verdict: str                            # Enjoy | Go easy | Limit | Avoid
    safe_grams: int                         # grams of this food safe to eat
    safe_pieces: str                        # human-friendly: "1 piece", "1-2 pieces", ...
    sugar_g: float                          # total sugar in safe_grams
    carbs_g: float                          # total carbs in safe_grams
    energy_kcal: float                      # total calories in safe_grams
    glycemic_load: float                    # the food's GL (for the "why" section)
    reasons: list[str]                      # 2-3 short, personalized bullets
    food_name: str
    festival: str
    region: str


class FoodInfo(BaseModel):
    """Food catalog entry returned by ``GET /foods``."""

    food_name: str
    festival: str
    region: str
    glycemic_index: float
    weight_g: float
    carbs_per_item_g: float
    sugar_per_item_g: float
    protein_per_item_g: float
    fat_per_item_g: float
    fiber_per_item_g: float
    energy_per_item_kcal: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
