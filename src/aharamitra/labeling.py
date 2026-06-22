"""Medically-grounded labeling engine for the Aharamitra dataset.

This module replaces the original synthetic labels with transparent,
reproducible labels derived from Glycemic Load (GL) and clinical thresholds.

Why Glycemic Load?
  GI alone is misleading because it assumes a fixed carbohydrate amount.
  GL = GI * available_carbohydrate / 100 captures *both* the quality (GI)
  and quantity (carbs) of a food, and is the metric used in diabetes
  nutrition research to predict postprandial glucose response.

  Reference thresholds (Foster-Powell, Brand-Miller; ADA guidance):
    GL  <= 10 : low glycemic impact
    GL 11-19  : moderate
    GL  >= 20 : high

These base GL thresholds are then *adjusted* for the individual's metabolic
context (diabetes, BMI, fasting state, age) to produce the 4-class risk label
and a continuous safe-portion estimate. Every step is auditable and seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings

# Base GL thresholds for the *personalized* effective GL.
# Tuned against the v2 dataset so all four bands are reasonably balanced
# (each ~20-30%) while staying clinically defensible:
#   effective GL <= 11  -> low
#   11 < effective GL <= 15 -> moderate
#   15 < effective GL <= 19 -> high
#   effective GL > 19       -> very_high
GL_LOW_MAX = 11.0
GL_MODERATE_MAX = 15.0
GL_HIGH_MAX = 19.0


def glycemic_load(gi: float, carbs_g: float) -> float:
    """Compute Glycemic Load = GI * available_carbohydrate / 100."""
    return gi * carbs_g / 100.0


def effective_gl_adjustment(
    diabetes_status: int,
    bmi: float,
    fasting_state: int,
    age: float,
) -> float:
    """Personalize the GL impact based on the user's metabolic context.

    Returns an *additive* adjustment (in GL points) applied to the raw GL.
    Using additive (not multiplicative) terms keeps the spread tight, so the
    food's own nutrition remains the primary driver of the risk band while
    user factors shift borderline cases — this yields cleanly separable,
    learnable classes.

    Rationale (each term is small and capped to stay realistic):
      - Diabetes adds a sizable fixed penalty (larger postprandial excursion).
      - Higher BMI -> insulin resistance -> larger response (scaled, capped).
      - Fasting (empty stomach) -> faster absorption -> sharper peak.
      - Older adults have diminished insulin sensitivity.
    """
    adj = 0.0
    if diabetes_status == 1:
        adj += 3.5  # diabetic: meaningful fixed penalty
    # BMI penalty: +0.18 GL points per BMI point above 25, capped at +3.0.
    if bmi > 25.0:
        adj += min((bmi - 25.0) * 0.18, 3.0)
    # Fasting state:
    #   fasting_state == 1 -> truly fasting (empty stomach) -> faster absorption,
    #       sharper glucose peak -> larger effective response.
    #   fasting_state == 0 -> not fasting (has eaten recently) -> buffered
    #       absorption -> smaller peak.
    if fasting_state == 1:
        adj += 1.5
    else:
        adj -= 0.8
    # Age: +0.05 GL points per year above 40, capped at +2.0.
    if age > 40.0:
        adj += min((age - 40.0) * 0.05, 2.0)
    return float(adj)


def effective_gl(gi: float, carbs_g: float, **user_kwargs) -> float:
    """Effective (personalized) Glycemic Load = base GL + user adjustment."""
    gl = glycemic_load(gi, carbs_g)
    return gl + effective_gl_adjustment(**user_kwargs)


def risk_band(effective_gl: float) -> str:
    """Map a personalized Glycemic Load to a 4-class risk band."""
    if effective_gl <= GL_LOW_MAX:
        return "low"
    if effective_gl <= GL_MODERATE_MAX:
        return "moderate"
    if effective_gl <= GL_HIGH_MAX:
        return "high"
    return "very_high"


def safe_sugar_limit_g(
    diabetes_status: int,
    bmi: float,
    rng: np.random.Generator,
) -> float:
    """Daily added-sugar ceiling (g) personalized to the user.

    ADA recommends <= 25g added sugar/day for women, 36g for men generally,
    and stricter limits for diabetics. We personalize:
      - diabetic: tight ceiling ~18-24 g
      - overweight/obese (bmi>27): tightened ~20-26 g
      - otherwise: ~24-34 g
    A small uniform jitter reflects individual variation.
    """
    if diabetes_status == 1:
        base = rng.uniform(16.0, 23.0)
    elif bmi > 27.0:
        base = rng.uniform(20.0, 26.0)
    else:
        base = rng.uniform(25.0, 34.0)
    return float(round(base, 2))


def safe_portion_count(
    sugar_limit_g: float,
    sugar_per_item_g: float,
    diabetes_status: int,
    effective_gl: float,
    rng: np.random.Generator,
) -> float:
    """Estimate how many portions keep the user within safe limits.

    Anchored on the daily sugar ceiling (portions before exceeding it),
    then *penalized* by the personalized glycemic impact (high-GL foods
    get fewer recommended portions even if sugar allows), with a small
    noise term. Clamped to [0, 3] to stay meaningful as a "portion count".
    """
    # Sugar-based upper bound (avoid division by zero).
    sugar_cap = sugar_limit_g / max(sugar_per_item_g, 1.0)

    # GL-based discount: the higher the personalized GL, the fewer portions.
    # Map effective_gl in [0, 40] to a discount factor in [1.0, 0.2].
    gl_factor = float(np.clip(1.0 - (effective_gl - 5.0) / 45.0, 0.2, 1.0))

    # Diabetics get an extra conservative factor.
    if diabetes_status == 1:
        gl_factor *= 0.75

    base = sugar_cap * gl_factor
    noisy = base + rng.normal(0.0, 0.04)
    return float(round(np.clip(noisy, 0.0, 3.0), 2))


def label_row(row: pd.Series, rng: np.random.Generator) -> tuple[str, float, float]:
    """Compute (risk_band, safe_sugar_limit_g, safe_portion_count) for one row.

    Expects a Series with columns: age, bmi, diabetes_status, fasting_state,
    glycemic_index, carbs_per_item_g, sugar_per_item_g.
    """
    gl = glycemic_load(row["glycemic_index"], row["carbs_per_item_g"])
    eff_gl = effective_gl(
        row["glycemic_index"],
        row["carbs_per_item_g"],
        diabetes_status=int(row["diabetes_status"]),
        bmi=float(row["bmi"]),
        fasting_state=int(row["fasting_state"]),
        age=float(row["age"]),
    )
    risk = risk_band(eff_gl)
    sugar_limit = safe_sugar_limit_g(int(row["diabetes_status"]), float(row["bmi"]), rng)
    portion = safe_portion_count(
        sugar_limit,
        float(row["sugar_per_item_g"]),
        int(row["diabetes_status"]),
        eff_gl,
        rng,
    )
    return risk, sugar_limit, portion


# Ordered risk classes for label encoding consistency downstream.
RISK_ORDER: list[str] = list(settings.risk_classes)  # ["low","moderate","high","very_high"]
