"""Centralized configuration for the Aharamitra project.

All paths, hyperparameters, and constants live here so nothing is hardcoded
across the codebase. Settings can be overridden via environment variables
(pydantic-settings).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root from this file: src/aharamitra/config.py -> ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project-wide settings.

    Values can be overridden through environment variables prefixed with
    ``AHARAMITRA_`` (e.g. ``AHARAMITRA_DATA_DIR=/tmp/data``).
    """

    model_config = SettingsConfigDict(
        env_prefix="AHARAMITRA_", env_file=".env", extra="ignore"
    )

    # --- Paths (relative to project root, overridable via env) ---------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # --- Datasets -----------------------------------------------------------
    original_dataset: str = "ritual_food_metabolic_dataset_original.csv"
    processed_dataset: str = "aharamitra_v2.csv"

    # --- Model artifacts ----------------------------------------------------
    classifier_artifact: str = "risk_classifier.joblib"
    regressor_artifact: str = "portion_regressor.joblib"
    metadata_artifact: str = "model_metadata.json"

    # --- Feature schema -----------------------------------------------------
    # Nominal categoricals -> OneHot encoded
    categorical_features: tuple[str, ...] = ("bmi_category", "festival", "region", "food_name")

    # Numeric features passed through (some engineered downstream)
    numeric_features: tuple[str, ...] = (
        "age",
        "bmi",
        "diabetes_status",
        "fasting_state",
        "glycemic_index",
        "carbs_per_item_g",
        "sugar_per_item_g",
        "protein_per_item_g",
        "fat_per_item_g",
        "fiber_per_item_g",
        "energy_per_item_kcal",
        "glycemic_load",
    )

    # Target columns
    classification_target: str = "glucose_spike_risk"
    regression_target: str = "safe_portion_count"

    # The four medically-grounded risk bands, ordered from low to high.
    risk_classes: tuple[str, ...] = ("low", "moderate", "high", "very_high")

    # --- Reproducibility ----------------------------------------------------
    random_state: int = 42

    # --- Training defaults --------------------------------------------------
    test_size: float = 0.2
    cv_folds: int = 5
    optuna_trials: int = 40

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.reports_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


# Module-level singleton for convenient import: `from aharamitra.config import settings`
settings = Settings()
