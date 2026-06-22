"""Build the enriched Aharamitra v2 dataset.

Generates a reproducible, clinically-grounded dataset by combining:
  1. Real per-food nutrition from the IFCT/USDA knowledge base (``foods.py``).
  2. Realistic user health profiles (age/BMI/diabetes/fasting distributions).
  3. Medically-grounded labels via the Glycemic-Load engine (``labeling.py``).

Run directly or via the CLI: ``python -m aharamitra.build_dataset``.

The original synthetic dataset is preserved in ``data/raw/`` for an honest
before/after comparison in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import settings
from .foods import FOODS
from .labeling import label_row

# Jitter magnitudes for per-sample nutrition variation (recipe/portion realism).
NUTRITION_JITTER = {
    "carbs_per_item_g": 0.08,
    "sugar_per_item_g": 0.10,
    "protein_per_item_g": 0.12,
    "fat_per_item_g": 0.12,
    "fiber_per_item_g": 0.15,
    "energy_per_item_kcal": 0.06,
    "glycemic_index": 0.04,
}


def _bmi_category(bmi: float) -> str:
    """Map a BMI value to its clinical category."""
    if bmi < 18.5:
        return "underweight"
    if bmi < 25.0:
        return "normal"
    if bmi < 30.0:
        return "overweight"
    return "obese"


def _sample_user(rng: np.random.Generator) -> dict:
    """Draw a single realistic user health profile.

    Distributions are chosen to mirror real population health data:
      - Age: uniform 18-80 (adult range).
      - BMI: mixture — most adults normal/overweight, a heavy obese tail.
      - Diabetes: ~11% prevalence (global adult estimate), higher when older/obese.
      - Fasting: ~50/50, biased slightly toward non-fasting.
    """
    age = int(rng.integers(18, 81))

    # BMI mixture: normal-ish core + overweight/obese skew.
    bmi = float(np.clip(rng.normal(loc=25.5, scale=5.0), 16.5, 42.0))

    # Diabetes probability rises with age and BMI.
    p_diabetes = 0.06 + 0.012 * max(0, age - 40) / 10.0 + 0.04 * (bmi > 27) + 0.06 * (bmi > 32)
    diabetes_status = int(rng.random() < min(p_diabetes, 0.5))

    fasting_state = int(rng.random() < 0.45)  # 1 = fasting

    return {
        "age": age,
        "bmi": round(bmi, 1),
        "bmi_category": _bmi_category(bmi),
        "diabetes_status": diabetes_status,
        "fasting_state": fasting_state,
    }


def _jitter_food(food: dict, rng: np.random.Generator) -> dict:
    """Add controlled per-portion jitter to a food's base nutrition."""
    out = dict(food)
    for key, scale in NUTRITION_JITTER.items():
        base = out[key]
        # Relative noise, never dropping below zero, rounded sensibly.
        noisy = base * (1.0 + rng.normal(0.0, scale))
        out[key] = round(max(noisy, 0.1), 2)
    return out


def build_dataset(n_rows: int = 20_000, seed: int | None = None) -> pd.DataFrame:
    """Generate the enriched dataset as a DataFrame.

    Each row pairs a realistic user with a (jittered) festival food and labels
    it with the GL engine. Food assignment is uniform across the catalogue so
    the model sees all foods; festival/region follow each food's provenance.
    """
    if seed is None:
        seed = settings.random_state
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    foods = list(FOODS)
    n_foods = len(foods)

    for _ in range(n_rows):
        user = _sample_user(rng)

        # Pick a food; mix same-region users with cross-region consumers
        # (festivals spread culturally), so region isn't trivially tied to food.
        food = _jitter_food(foods[rng.integers(0, n_foods)], rng)

        row = {**user}
        row["festival"] = food["festival"]
        # 70% of the time the consumer is from the food's home region,
        # otherwise a random other region (cultural spread).
        all_regions = sorted({f["region"] for f in foods})
        if rng.random() < 0.7:
            row["region"] = food["region"]
        else:
            choices = [r for r in all_regions if r != food["region"]] or all_regions
            row["region"] = choices[rng.integers(0, len(choices))]

        row["food_name"] = food["food_name"]
        row["glycemic_index"] = food["glycemic_index"]
        row["carbs_per_item_g"] = food["carbs_per_item_g"]
        row["sugar_per_item_g"] = food["sugar_per_item_g"]
        row["protein_per_item_g"] = food["protein_per_item_g"]
        row["fat_per_item_g"] = food["fat_per_item_g"]
        row["fiber_per_item_g"] = food["fiber_per_item_g"]
        row["energy_per_item_kcal"] = food["energy_per_item_kcal"]

        # Engineered feature: glycemic load (also used internally for labeling).
        from .labeling import glycemic_load

        row["glycemic_load"] = round(
            glycemic_load(row["glycemic_index"], row["carbs_per_item_g"]), 2
        )

        # Label with the medical engine.
        risk, sugar_limit, portion = label_row(pd.Series(row), rng)
        row["glucose_spike_risk"] = risk
        row["safe_sugar_limit_g"] = sugar_limit
        row["safe_portion_count"] = portion

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main(n_rows: int = 20_000) -> None:
    """Build and persist the dataset to ``data/processed/``."""
    settings.ensure_dirs()
    df = build_dataset(n_rows=n_rows)

    out_path = settings.processed_data_dir / settings.processed_dataset
    df.to_csv(out_path, index=False)

    print(f"✓ Dataset written: {out_path}")
    print(f"  rows={len(df)}  cols={len(df.columns)}")
    print("\nRisk distribution:")
    print(df["glucose_spike_risk"].value_counts().sort_index().to_string())
    print("\nPer-food counts (top):")
    print(df["food_name"].value_counts().head(5).to_string())


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def build(n_rows: int = 20_000, seed: int = settings.random_state):
        """Build the enriched dataset (default 20,000 rows)."""
        import aharamitra.config as cfg

        cfg.settings.random_state = seed
        main(n_rows=n_rows)

    app()
