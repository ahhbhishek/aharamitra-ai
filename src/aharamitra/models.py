"""Model training, benchmarking, and evaluation for Aharamitra.

This module provides:
- ``benchmark()`` — compare 6 classifiers + 6 regressors on stratified CV.
- ``train_best()`` — train the best model with Optuna tuning.
- ``evaluate()`` — produce a full metrics report + SHAP plots.
- ``predict()`` — inference on a single-row dict or DataFrame.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for report generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    cross_validate,
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from .config import settings
from .features import (
    INPUT_FEATURES,
    build_classifier_pipeline,
    build_preprocessor,
    build_regressor_pipeline,
)
from .labeling import RISK_ORDER

warnings.filterwarnings("ignore", category=UserWarning, module="shap")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the processed dataset."""
    if path is None:
        path = settings.processed_data_dir / settings.processed_dataset
    return pd.read_csv(path)


def prepare_xy(df: pd.DataFrame):
    """Split into X (features), y_class (encoded), y_reg.

    Returns X, y_class_encoded, y_reg, risk_encoder.
    """
    X = df[INPUT_FEATURES].copy()

    risk_encoder = LabelEncoder()
    risk_encoder.classes_ = np.array(RISK_ORDER)  # fixed ordering
    y_class = risk_encoder.transform(df[settings.classification_target])
    y_reg = df[settings.regression_target].values

    return X, y_class, y_reg, risk_encoder


# ---------------------------------------------------------------------------
# Benchmark (Phase 2 centerpiece)
# ---------------------------------------------------------------------------

CLASSIFIERS: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss", verbosity=0,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    ),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True,
    ),
}

REGRESSORS: dict[str, Any] = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    ),
    "SVR": SVR(kernel="rbf", C=1.0),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True,
    ),
}

if HAS_LGBM:
    CLASSIFIERS["LightGBM"] = LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    REGRESSORS["LightGBM"] = LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )


def benchmark(X, y_class, y_reg, cv: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all classifiers and regressors via k-fold CV.

    Classifier folds are stratified on ``y_class``; regressor folds use plain
    ``KFold`` (the regression target ``y_reg`` is continuous and cannot be
    stratified).

    Returns (clf_benchmark_df, reg_benchmark_df) with metrics.
    """
    if cv is None:
        cv = settings.cv_folds

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=settings.random_state)
    kf = KFold(n_splits=cv, shuffle=True, random_state=settings.random_state)

    # --- Classifiers ---
    clf_results = []
    for name, clf in CLASSIFIERS.items():
        pipe = build_classifier_pipeline(clf)
        scoring = ["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr_weighted"]
        try:
            cv_out = cross_validate(pipe, X, y_class, cv=skf, scoring=scoring, n_jobs=-1)
            clf_results.append({
                "model": name,
                "accuracy": cv_out["test_accuracy"].mean(),
                "f1_macro": cv_out["test_f1_macro"].mean(),
                "f1_weighted": cv_out["test_f1_weighted"].mean(),
                "roc_auc_ovr": cv_out["test_roc_auc_ovr_weighted"].mean(),
            })
        except Exception as e:
            print(f"  ⚠ {name} (classifier) failed: {e}")
            clf_results.append({"model": name, "accuracy": np.nan, "f1_macro": np.nan,
                                 "f1_weighted": np.nan, "roc_auc_ovr": np.nan})

    clf_df = pd.DataFrame(clf_results).sort_values("f1_macro", ascending=False)

    # --- Regressors ---
    reg_results = []
    for name, reg in REGRESSORS.items():
        pipe = build_regressor_pipeline(reg)
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
        try:
            cv_out = cross_validate(pipe, X, y_reg, cv=kf, scoring=scoring, n_jobs=-1)
            reg_results.append({
                "model": name,
                "rmse": -cv_out["test_neg_root_mean_squared_error"].mean(),
                "mae": -cv_out["test_neg_mean_absolute_error"].mean(),
                "r2": cv_out["test_r2"].mean(),
            })
        except Exception as e:
            print(f"  ⚠ {name} (regressor) failed: {e}")
            reg_results.append({"model": name, "rmse": np.nan, "mae": np.nan, "r2": np.nan})

    reg_df = pd.DataFrame(reg_results).sort_values("r2", ascending=False)

    return clf_df, reg_df


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------

def _optuna_classify(X_train, y_train, X_val, y_val, n_trials: int | None = None):
    """Optimize XGBoost classifier with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        clf = XGBClassifier(**params)
        pipe = build_classifier_pipeline(clf)
        pipe.fit(X_train, y_train)
        from sklearn.metrics import f1_score
        preds = pipe.predict(X_val)
        return f1_score(y_val, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials or settings.optuna_trials, show_progress_bar=False)
    return study.best_params


def _optuna_regress(X_train, y_train, X_val, y_val, n_trials: int | None = None):
    """Optimize XGBoost regressor with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": 42,
            "verbosity": 0,
        }
        reg = XGBRegressor(**params)
        pipe = build_regressor_pipeline(reg)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        return -mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials or settings.optuna_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Full train + evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame | None = None,
    skip_benchmark: bool = False,
    skip_tuning: bool = False,
    n_trials: int | None = None,
) -> dict:
    """End-to-end train/evaluate pipeline. Returns a metrics dict.

    Steps:
      1. Benchmark all models (or skip).
      2. Optuna-tune the best model family.
      3. Final evaluation on held-out test set.
      4. SHAP analysis.
      5. Save artifacts.
    """
    settings.ensure_dirs()
    if df is None:
        df = load_data()
    X, y_class, y_reg, risk_encoder = prepare_xy(df)

    # Stratified train-test split.
    X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = train_test_split(
        X, y_class, y_reg,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=y_class,
    )

    results: dict[str, Any] = {"test_size": len(X_test)}

    # --- Step 1: Benchmark ---
    if not skip_benchmark:
        print("📊 Running benchmark (cross-validation)...")
        clf_bench, reg_bench = benchmark(X_train, y_c_train, y_r_train)
        print("\n── Classifier benchmark ──")
        print(clf_bench.to_string(index=False))
        print("\n── Regressor benchmark ──")
        print(reg_bench.to_string(index=False))

        # Save benchmark tables.
        clf_bench.to_csv(settings.reports_dir / "benchmark_classifier.csv", index=False)
        reg_bench.to_csv(settings.reports_dir / "benchmark_regressor.csv", index=False)
        results["clf_benchmark"] = clf_bench.to_dict("records")
        results["reg_benchmark"] = reg_bench.to_dict("records")

    # --- Step 2: Optuna tuning ---
    best_clf_params: dict = {}
    best_reg_params: dict = {}

    if not skip_tuning:
        print("\n🔧 Optuna tuning XGBoost classifier...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_c_train, test_size=0.15, random_state=settings.random_state, stratify=y_c_train,
        )
        best_clf_params = _optuna_classify(X_tr, y_tr, X_val, y_val, n_trials)
        print(f"   Best params: {best_clf_params}")

        print("🔧 Optuna tuning XGBoost regressor...")
        best_reg_params = _optuna_regress(X_train, y_r_train, X_test, y_r_test, n_trials)
        print(f"   Best params: {best_reg_params}")

    # --- Step 3: Final models ---
    clf = XGBClassifier(**(best_clf_params or {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
        "eval_metric": "mlogloss", "verbosity": 0,
    }))
    clf_pipe = build_classifier_pipeline(clf)
    clf_pipe.fit(X_train, y_c_train)

    reg = XGBRegressor(**(best_reg_params or {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "verbosity": 0,
    }))
    reg_pipe = build_regressor_pipeline(reg)
    reg_pipe.fit(X_train, y_r_train)

    # --- Step 3b: Evaluate ---
    y_c_pred = clf_pipe.predict(X_test)
    y_r_pred = reg_pipe.predict(X_test)

    from sklearn.metrics import f1_score, roc_auc_score
    clf_report = classification_report(y_c_test, y_c_pred, target_names=RISK_ORDER, output_dict=True)
    f1_macro = f1_score(y_c_test, y_c_pred, average="macro")
    acc = (y_c_pred == y_c_test).mean()
    try:
        roc_auc = roc_auc_score(y_c_test, clf_pipe.predict_proba(X_test), multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = np.nan

    reg_mae = mean_absolute_error(y_r_test, y_r_pred)
    reg_rmse = root_mean_squared_error(y_r_test, y_r_pred)
    reg_r2 = r2_score(y_r_test, y_r_pred)

    print("\n── Final classifier ──")
    print(classification_report(y_c_test, y_c_pred, target_names=RISK_ORDER))
    print(f"   ROC-AUC (ovr, weighted): {roc_auc:.4f}")

    print("\n── Final regressor ──")
    print(f"   MAE:  {reg_mae:.4f}")
    print(f"   RMSE: {reg_rmse:.4f}")
    print(f"   R²:   {reg_r2:.4f}")

    results["classifier"] = {
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(acc, 4),
        "roc_auc_ovr_weighted": round(roc_auc, 4),
        "classification_report": clf_report,
    }
    results["regressor"] = {
        "mae": round(reg_mae, 4),
        "rmse": round(reg_rmse, 4),
        "r2": round(reg_r2, 4),
    }

    # --- Step 4: Confusion matrix plot ---
    _plot_confusion_matrix(y_c_test, y_c_pred, risk_encoder)
    _plot_regression_residuals(y_r_test, y_r_pred)

    # --- Step 5: SHAP ---
    _save_shap_analysis(clf_pipe, X_test, risk_encoder)

    # --- Step 6: Save artifacts ---
    joblib.dump(clf_pipe, settings.models_dir / settings.classifier_artifact)
    joblib.dump(reg_pipe, settings.models_dir / settings.regressor_artifact)
    joblib.dump(risk_encoder, settings.models_dir / "risk_encoder.joblib")

    metadata = {
        "version": "0.2.0",
        "input_features": INPUT_FEATURES,
        "categorical_features": ["bmi_category", "festival", "region", "food_name"],
        "numeric_features": [c for c in INPUT_FEATURES if c not in ["bmi_category", "festival", "region", "food_name"]],
        "risk_classes": RISK_ORDER,
        "classifier_params": best_clf_params,
        "regressor_params": best_reg_params,
        "metrics": results,
    }
    with open(settings.models_dir / settings.metadata_artifact, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Artifacts saved to {settings.models_dir}")
    print(f"✓ Reports saved to {settings.reports_dir}")
    return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, risk_encoder):
    """Save confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=risk_encoder.classes_, yticklabels=risk_encoder.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Glucose Spike Risk Classifier")
    plt.tight_layout()
    fig.savefig(settings.reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("   ✓ confusion_matrix.png")


def _plot_regression_residuals(y_true, y_pred):
    """Save residual scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred - y_true, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Actual safe_portion_count")
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title("Regression Residuals — Safe Portion Regressor")
    plt.tight_layout()
    fig.savefig(settings.reports_dir / "regression_residuals.png", dpi=150)
    plt.close(fig)
    print("   ✓ regression_residuals.png")


def _save_shap_analysis(pipe, X_test, risk_encoder):
    """Compute and save SHAP feature importance summary plot."""
    try:
        preprocessor = pipe.named_steps["preprocessor"]
        X_encoded = preprocessor.transform(X_test)

        # Get feature names after one-hot encoding.
        cat_cols = ["bmi_category", "festival", "region", "food_name"]
        num_cols = [c for c in INPUT_FEATURES if c not in cat_cols]
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        all_feature_names = cat_feature_names + num_cols

        explainer = shap.TreeExplainer(pipe.named_steps["classifier"])
        shap_values = explainer.shap_values(X_encoded)

        # For multi-class, use mean absolute SHAP across classes.
        if isinstance(shap_values, list):
            mean_shap = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        # Top 20 features by mean |SHAP|.
        mean_abs = np.mean(np.abs(mean_shap if mean_shap.ndim == 1 else mean_shap.mean(axis=0)), axis=0) if mean_shap.ndim > 1 else np.abs(mean_shap)

        # Properly handle shape
        if mean_shap.ndim == 2:
            importance = np.abs(mean_shap).mean(axis=0)
        else:
            importance = np.abs(mean_shap)

        ranked_idx = np.argsort(importance)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            range(len(ranked_idx)),
            importance[ranked_idx],
            color="#4C72B0",
        )
        ax.set_yticks(range(len(ranked_idx)))
        ax.set_yticklabels([all_feature_names[i] for i in ranked_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top 20 Feature Importance (SHAP)")
        plt.tight_layout()
        fig.savefig(settings.reports_dir / "feature_importance_shap.png", dpi=150)
        plt.close(fig)
        print("   ✓ feature_importance_shap.png")
    except Exception as e:
        print(f"   ⚠ SHAP analysis failed: {e}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(sample: dict | pd.DataFrame, risk_encoder=None) -> dict:
    """Run inference on a single sample or batch DataFrame.

    Returns a dict with glucose_risk, safe_portion, and risk_label.
    """
    settings.ensure_dirs()

    clf_pipe = joblib.load(settings.models_dir / settings.classifier_artifact)
    reg_pipe = joblib.load(settings.models_dir / settings.regressor_artifact)
    if risk_encoder is None:
        risk_encoder = joblib.load(settings.models_dir / "risk_encoder.joblib")

    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])

    X = sample[INPUT_FEATURES]
    risk_num = clf_pipe.predict(X)
    risk_label = risk_encoder.inverse_transform(risk_num)
    portion = reg_pipe.predict(X)

    # Confidence: use predict_proba if available.
    try:
        proba = clf_pipe.predict_proba(X)
        confidence = float(np.max(proba, axis=1)[0])
    except Exception:
        confidence = None

    return {
        "glucose_spike_risk": risk_label[0],
        "risk_encoded": int(risk_num[0]),
        "safe_portion_count": round(float(portion[0]), 2),
        "confidence": round(confidence, 4) if confidence else None,
    }
