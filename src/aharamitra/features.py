"""Preprocessing pipeline for Aharamitra.

Builds a single ``sklearn.Pipeline`` with a ``ColumnTransformer`` so that
training and inference can *never* diverge.  Categorical columns get
``OneHotEncoder(handle_unknown='ignore')``, numeric columns pass through.

Drop ``bmi_category`` (fully derivable from ``bmi`` — redundant/leaky).
Drop ``safe_sugar_limit_g`` (only used as an intermediate in labeling;
not an input feature).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Columns the model reads as input.
INPUT_FEATURES: list[str] = [
    "age",
    "bmi",
    "diabetes_status",
    "fasting_state",
    "bmi_category",
    "festival",
    "region",
    "food_name",
    "glycemic_index",
    "carbs_per_item_g",
    "sugar_per_item_g",
    "protein_per_item_g",
    "fat_per_item_g",
    "fiber_per_item_g",
    "energy_per_item_kcal",
    "glycemic_load",
]

# Nominal categoricals → OneHot encoded.
CATEGORICAL_COLS: list[str] = [
    "bmi_category",
    "festival",
    "region",
    "food_name",
]

# Everything else in INPUT_FEATURES that isn't categorical.
NUMERIC_COLS: list[str] = [c for c in INPUT_FEATURES if c not in CATEGORICAL_COLS]


def build_preprocessor() -> ColumnTransformer:
    """Return the ``ColumnTransformer`` that encodes raw DataFrames.

    Categorical columns  →  ``OneHotEncoder(sparse_output=False,
                         handle_unknown='ignore')``
    Numeric columns      →  passthrough
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop",  # drop any column not explicitly listed
    )


def build_classifier_pipeline(classifier) -> Pipeline:
    """Wrap *classifier* with the shared preprocessor."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def build_regressor_pipeline(regressor) -> Pipeline:
    """Wrap *regressor* with the shared preprocessor."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", regressor),
    ])
