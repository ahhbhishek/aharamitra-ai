"""Pydantic request/response schemas for the Aharamitra API.

These define the contract between the React UI and the FastAPI backend,
and enforce input validation at the boundary.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# Allowed categorical values (kept in sync with the food knowledge base).
BMI_CATEGORIES = ("underweight", "normal", "overweight", "obese")


class PredictionRequest(BaseModel):
    """Inbound prediction request matching the model's input schema."""

    age: int = Field(..., ge=18, le=100, description="Patient age in years")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    diabetes_status: int = Field(..., ge=0, le=1, description="1 = diabetic, 0 = non-diabetic")
    fasting_state: int = Field(..., ge=0, le=1, description="1 = fasting, 0 = not fasting")
    bmi_category: str = Field(..., description="BMI category label")
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

    @field_validator("bmi_category")
    @classmethod
    def _valid_bmi_category(cls, v: str) -> str:
        if v not in BMI_CATEGORIES:
            raise ValueError(f"bmi_category must be one of {BMI_CATEGORIES}")
        return v

    def to_model_input(self) -> dict:
        """Convert to a dict suitable for the model, computing glycemic_load."""
        from aharamitra.labeling import glycemic_load

        d = self.model_dump()
        d["glycemic_load"] = round(glycemic_load(self.glycemic_index, self.carbs_per_item_g), 2)
        return d


class PredictionResponse(BaseModel):
    """Outbound prediction result."""

    glucose_spike_risk: str
    risk_encoded: int
    safe_portion_count: float
    confidence: Optional[float] = None
    food_name: str
    festival: str
    region: str


class FoodInfo(BaseModel):
    """Food catalog entry returned by ``GET /foods``."""

    food_name: str
    festival: str
    region: str
    glycemic_index: float
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
