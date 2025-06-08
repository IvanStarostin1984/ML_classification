# TODO: Modularise the Colab notebook

The repository at commit dbd5184 only contains the legacy `ai_arisha.py` script, `README.md`, `AGENTS.md` and a dataset README.  The goal is to reorganise the notebook code into a maintainable module-based project with CI tests.

## 1. Basic project skeleton
- [x] create directories: `.github/workflows/`, `src/models/`, `scripts/`, `tests/`, and `notebooks/`
- [x] add minimal files listed in `AGENTS.md` and `README.md`: `environment.yml`, `requirements.txt`, `Dockerfile`, `Makefile`, `LICENSE`, `.gitignore`

## 2. Data utilities
- implement `scripts/download_data.py` to fetch the Kaggle dataset using `KAGGLE_USERNAME` and `KAGGLE_KEY`
- write `src/dataprep.py` to load the raw CSV and perform basic cleaning

## 3. Feature engineering
- move the lengthy feature engineering block from `ai_arisha.py` into `src/features.py`
- split out diagnostic plots and chi-square analysis into `src/diagnostics.py`
- provide preprocessing helpers under `src/preprocessing.py` and feature selection logic under `src/selection.py`

## 4. Modelling pipelines
- add `src/models/logreg.py` and `src/models/cart.py` for logistic regression and decision-tree pipelines respectively
- create `src/split.py` with stratified train/validation/test split utilities
- expose a simple command line entry point (e.g. `make train` or `python -m src.models.logreg`)

## 5. Tests and CI
- [x] add `tests/test_smoke.py` importing each module
- [x] set up GitHub Actions workflow `ci.yml` running flake8/black and `pytest`
- add unit tests for `dataprep`, `features`, and `models` modules

## 6. Documentation updates
- update `README.md` with new instructions once modules are in place
- add brief usage notes to `notebooks/README.md`

## 7. Legacy script
- keep `ai_arisha.py` read-only for reference until the migration is finished
