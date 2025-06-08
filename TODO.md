# TODO: Modularise the Colab notebook

All migration tasks are complete as of commit `8af97fc`. This checklist started from commit `dbd5184` when only the legacy `ai_arisha.py` and a few docs were present. The project now has modular code under `src/`, a suite of tests and CI workflow.

## 1. Basic project skeleton
- [x] create directories: `.github/workflows/`, `src/models/`, `scripts/`, `tests/`, and `notebooks/`
- [x] add minimal files listed in `AGENTS.md` and `README.md`: `environment.yml`, `requirements.txt`, `Dockerfile`, `Makefile`, `LICENSE`, `.gitignore`

- [x] implement `scripts/download_data.py` to fetch the Kaggle dataset using `KAGGLE_USERNAME` and `KAGGLE_KEY`
- [x] write `src/dataprep.py` to load the raw CSV and perform basic cleaning

## 3. Feature engineering
- [x] move the lengthy feature engineering block from `ai_arisha.py` into `src/features.py`
- [x] split out diagnostic plots and chi-square analysis into `src/diagnostics.py`
- [x] provide preprocessing helpers under `src/preprocessing.py` and feature selection logic under `src/selection.py`

## 4. Modelling pipelines
- [x] add `src/models/logreg.py` and `src/models/cart.py` for logistic regression and decision-tree pipelines respectively
- [x] create `src/split.py` with stratified train/validation/test split utilities
- [x] expose a simple command line entry point (e.g. `make train` or `python -m src.models.logreg`)

## 5. Tests and CI
- [x] add `tests/test_smoke.py` importing each module
- [x] set up GitHub Actions workflow `ci.yml` running flake8/black and `pytest`
- [x] add unit tests for `dataprep`, `features`, and `models` modules

## 6. Documentation updates
- [x] update `README.md` with new instructions once modules are in place
- [x] remove note about missing model pipelines once they are added
    (obsolete since README now describes both pipelines)
- [x] add brief usage notes to `notebooks/README.md`

## 7. Legacy script
- keep `ai_arisha.py` read-only for reference until the migration is finished

## 8. Evaluation & fairness
- [x] implement evaluation CLI and fairness metrics
