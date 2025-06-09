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


##Missing elements from the modularised project

Inspection of ai_arisha.py reveals several features that were not ported to the src/ modules:


Extensive hyper‑parameter grids

ai_arisha.py defines larger parameter grids for both logistic regression and decision tree models (e.g. varying C, penalty, class_weight, tree depth, leaf size). The modular code has minimal grids of two values for each model.
Grid-search flag now calls these grids from the CLI.
ai_arisha.py defines larger parameter grids for both logistic regression and decision tree models (e.g. varying C, penalty, class_weight, tree depth, leaf size). The logistic grid is now exposed via ``grid_train_from_df`` but the tree model still uses a minimal grid.


Repeated cross‑validation and bootstrap logic
The original script uses RepeatedStratifiedKFold and falls back to bootstrapping when the minority class is small, recording confidence intervals over folds. The modular code runs a single 3×3 nested CV without bootstrapping.

Oversampling options, probability calibration, feature importance export, extended metrics and manifest writing were implemented in commit `0c16cae`.

## 9. Usability improvements
- [x] download_data prints guidance if src package cannot be imported.
 - [x] Clarify that `make` is needed for training commands and mention console scripts for Windows.


## 10. Modelling improvements
- [x] Add --grid-search option for repeated cross-validation and extended parameter grids.

- [x] port grid search helper for decision tree

- [x] save best estimator when performing cart grid search

- [x] Verify that each function from ai_arisha.py is represented or intentionally omitted in the src modules (see FUNCTIONS.md).

- [x] port reporting helpers for final artifact collection

- [x] add evaluation_utils helpers for plotting and fairness aliases


- [x] add tests for calibrate_model isotonic option and invalid method handling

- [x] add CLI test for sampler option

- [x] extend FeatureEngineer unit tests for column normalisation, asset ratios and risk flag


