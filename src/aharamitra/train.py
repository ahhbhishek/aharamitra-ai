"""CLI for the Aharamitra project.

Entry points:

    python -m aharamitra.train          # build dataset + full train + evaluate
    python -m aharamitra.train train    # (same, explicit)
    python -m aharamitra.train predict  # single-sample inference

``python -m aharamitra.train`` defaults to the ``train`` command for convenience.
"""

from __future__ import annotations

import sys

import typer

app = typer.Typer(add_completion=False, help="Aharamitra ML CLI")


@app.command()
def train(
    skip_benchmark: bool = typer.Option(False, "--skip-benchmark", help="Skip the CV benchmark table"),
    skip_tuning: bool = typer.Option(False, "--skip-tuning", help="Skip Optuna tuning (use defaults)"),
    n_trials: int = typer.Option(40, "--trials", "-n", help="Number of Optuna trials"),
    n_rows: int = typer.Option(20_000, "--rows", help="Dataset size to generate"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Build the dataset and train/evaluate all models."""
    import aharamitra.config as cfg

    cfg.settings.random_state = seed

    # Step 1: build dataset
    typer.echo("📦 Building dataset...")
    from .build_dataset import build_dataset

    df = build_dataset(n_rows=n_rows, seed=seed)
    df.to_csv(cfg.settings.processed_data_dir / cfg.settings.processed_dataset, index=False)
    typer.echo(f"   {len(df)} rows, {len(df.columns)} columns")

    # Step 2: train + evaluate
    typer.echo("🚀 Training models...")
    from .models import train_and_evaluate

    results = train_and_evaluate(
        df=df,
        skip_benchmark=skip_benchmark,
        skip_tuning=skip_tuning,
        n_trials=n_trials,
    )

    # Summary
    clf = results.get("classifier", {})
    reg = results.get("regressor", {})
    typer.echo(f"\n{'='*50}")
    typer.echo(f"  Classifier  accuracy: {clf.get('accuracy', '?')}")
    typer.echo(f"  Classifier  f1_macro: {clf.get('f1_macro', '?')}")
    typer.echo(f"  Classifier  ROC-AUC:  {clf.get('roc_auc_ovr_weighted', '?')}")
    typer.echo(f"  Regressor   MAE:      {reg.get('mae', '?')}")
    typer.echo(f"  Regressor   R²:       {reg.get('r2', '?')}")
    typer.echo(f"{'='*50}")


@app.command()
def predict(
    age: int = typer.Option(45, help="Patient age"),
    bmi: float = typer.Option(27.5, help="BMI"),
    diabetes_status: int = typer.Option(1, help="1=diabetic, 0=non-diabetic"),
    fasting_state: int = typer.Option(0, help="1=fasting, 0=not fasting"),
    bmi_category: str = typer.Option("overweight"),
    festival: str = typer.Option("Ganesh Chaturthi"),
    region: str = typer.Option("Maharashtra"),
    food_name: str = typer.Option("Steamed Modak"),
    glycemic_index: float = typer.Option(60),
    carbs_per_item_g: float = typer.Option(20),
    sugar_per_item_g: float = typer.Option(10),
    protein_per_item_g: float = typer.Option(2.0),
    fat_per_item_g: float = typer.Option(3.0),
    fiber_per_item_g: float = typer.Option(0.8),
    energy_per_item_kcal: float = typer.Option(120),
):
    """Run a single prediction using saved models."""
    from .models import predict as _predict
    from .labeling import glycemic_load

    gl = glycemic_load(glycemic_index, carbs_per_item_g)

    sample = {
        "age": age,
        "bmi": bmi,
        "diabetes_status": diabetes_status,
        "fasting_state": fasting_state,
        "bmi_category": bmi_category,
        "festival": festival,
        "region": region,
        "food_name": food_name,
        "glycemic_index": glycemic_index,
        "carbs_per_item_g": carbs_per_item_g,
        "sugar_per_item_g": sugar_per_item_g,
        "protein_per_item_g": protein_per_item_g,
        "fat_per_item_g": fat_per_item_g,
        "fiber_per_item_g": fiber_per_item_g,
        "energy_per_item_kcal": energy_per_item_kcal,
        "glycemic_load": round(gl, 2),
    }

    result = _predict(sample)
    typer.echo(f"\n🍽️  Food: {food_name} ({festival}, {region})")
    typer.echo(f"👤 User: age={age}, BMI={bmi}, diabetes={diabetes_status}, fasting={fasting_state}")
    typer.echo(f"\n   ⚠️  Glucose risk: {result['glucose_spike_risk'].upper()}")
    typer.echo(f"   🍽️  Safe portion:  {result['safe_portion_count']} servings")
    if result["confidence"]:
        typer.echo(f"   📊  Confidence:   {result['confidence']*100:.1f}%")


def _default_command(argv: list[str]) -> list[str]:
    """If no subcommand is given, default to ``train``."""
    # Skip flags-only invocation: ['--seed','7'] -> ['train','--seed','7']
    if not argv or argv[0].startswith("-"):
        return ["train", *argv]
    return argv


if __name__ == "__main__":
    app(_default_command(sys.argv[1:]))
