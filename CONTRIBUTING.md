# Contributing to Aharamitra

Thank you for your interest in contributing! This guide covers the basics.

## Development Setup

```bash
git clone https://github.com/ahhbhishek/aharamitra-ai.git
cd aharamitra-ai

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -e ".[api,dev]"
make data
make train
```

## Code Style

- **Python**: Black (line-length 100) + Ruff + isort
- **TypeScript/React**: Vite + Tailwind defaults (no Prettier config needed)
- Run `make test` before pushing

## Making Changes

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes, add tests where applicable
3. Run `make test` to verify nothing is broken
4. Commit with descriptive messages: `git commit -m "Add foo for bar scenario"`
5. Push and open a Pull Request

## Project Structure

See [README.md](README.md#project-structure) for the full layout.

Key modules:
- `src/aharamitra/foods.py` — add new foods here (follow the `FoodNutrition` TypedDict)
- `src/aharamitra/labeling.py` — risk band thresholds and GL adjustment logic
- `api/main.py` — add new API endpoints here
- `ui/src/App.tsx` — React UI components

## Adding a New Food

1. Open `src/aharamitra/foods.py`
2. Add a new `FoodNutrition` entry with real nutritional data (source: IFCT or USDA)
3. Run `make data` to regenerate the dataset
4. Run `make train` to retrain models
5. Verify the new food appears in `/foods` endpoint and UI dropdown

## Adding a New ML Model

1. Open `src/aharamitra/models.py`
2. Add the model to `CLASSIFIERS` or `REGRESSORS` dict
3. Run `make train` — it will automatically benchmark the new model
4. If it outperforms the current best, update the default model in the pipeline

## Reporting Issues

Please use [GitHub Issues](https://github.com/ahhbhishek/aharamitra-ai/issues) with:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
