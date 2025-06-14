# TODO: Modularise the Colab notebook

All migration tasks are complete as of commit `8af97fc`. This checklist started
from commit `dbd5184` when only the legacy `ai_arisha.py` and a few docs were
present. The project now has modular code under `src/`, a suite of tests and CI
workflow.

## 1. Basic project skeleton

- [x] create directories: `.github/workflows/`, `src/models/`, `scripts/`,
`tests/`, and `notebooks/`
- [x] add minimal files listed in `AGENTS.md` and `README.md`:
`environment.yml`, `requirements.txt`, `Dockerfile`, `Makefile`, `LICENSE`,
`.gitignore`

- [x] implement `scripts/download_data.py` to fetch the Kaggle dataset using
`KAGGLE_USERNAME` and `KAGGLE_KEY`
- [x] write `src/dataprep.py` to load the raw CSV and perform basic cleaning

## 3. Feature engineering

- [x] move the lengthy feature engineering block from `ai_arisha.py` into
`src/features.py`
- [x] split out diagnostic plots and chi-square analysis into
`src/diagnostics.py`
- [x] provide preprocessing helpers under `src/preprocessing.py` and feature
selection logic under `src/selection.py`

## 4. Modelling pipelines

- [x] add `src/models/logreg.py` and `src/models/cart.py` for logistic
regression and decision-tree pipelines respectively
- [x] create `src/split.py` with stratified train/validation/test split
utilities
- [x] expose a simple command line entry point (e.g. `make train` or `python -m
src.models.logreg`)

## 5. Tests and CI

- [x] add `tests/test_smoke.py` importing each module
- [x] set up GitHub Actions workflow `ci.yml` running flake8/black and `pytest`
- [x] add unit tests for `dataprep`, `features`, and `models` modules
- [x] add unit tests for `split.stratified_split`
- [x] add docs-only CI job running markdownlint and markdown-link-check
- [x] fix link check step to iterate over markdown files with find/xargs
- [x] Add pre-commit hook or make target with `npx markdownlint-cli` for MD012.
- [x] Run `pre-commit run --files` in CI before flake8, black and pytest

## 6. Documentation updates

- [x] update `README.md` with new instructions once modules are in place
- [x] remove note about missing model pipelines once they are added
      (obsolete since README now describes both pipelines)
- [x] add brief usage notes to `notebooks/README.md`
- [x] refresh README layout with new modules and replace docker compose mention
- [x] keep `AGENTS.md` project structure entries in sync with code and tests
- [x] document Markdown style guidelines in `AGENTS.md`
- [x] ensure markdown files pass `markdownlint`
- [x] fix NOTES.md long lines to satisfy markdownlint
- [x] shorten long NOTES lines to satisfy markdownlint
- [x] expand Sphinx docs with module usage and CLI examples
- [x] list Sphinx in requirements and update build instructions
- [x] fix README ROC-AUC badge lines via reference link
- [x] document single blank line rule in AGENTS and ignore actions URL in
      link checker

## 7. Legacy script

- keep `ai_arisha.py` read-only for reference until the migration is finished

## 8. Evaluation & fairness

- [x] implement evaluation CLI and fairness metrics

## Missing elements from the modularised project

Inspection of ai_arisha.py reveals several features that were not ported to the
src/ modules:

Extensive hyper‑parameter grids

ai_arisha.py defines larger parameter grids for both logistic regression and
decision tree models (e.g. varying C, penalty, class_weight, tree depth, leaf
size). The modular code has minimal grids of two values for each model.
Grid-search flag now calls these grids from the CLI.
ai_arisha.py defines larger parameter grids for both logistic regression and
decision tree models (e.g. varying C, penalty, class_weight, tree depth, leaf
size). The logistic grid is now exposed via `grid_train_from_df` but the tree
model still uses a minimal grid.

Repeated cross‑validation and bootstrap logic
The original script uses RepeatedStratifiedKFold and falls back to
bootstrapping when the minority class is small, recording confidence intervals
over folds. The modular code runs a single 3×3 nested CV without bootstrapping.

Oversampling options, probability calibration, feature importance export,
extended metrics and manifest writing were implemented in commit `0c16cae`.

- [x] Ported statistical helpers `_need_exact`, `_cramers_v`,
`_cochran_armitage`,
      `_safe_chi2`, and `_fmt_p`/`_annotate` into new
      `src/diagnostics_stats.py` with unit tests.

## 9. Usability improvements

- [x] download_data prints guidance if src package cannot be imported.
- [x] Clarify that `make` is needed for training commands and mention console
scripts for Windows.

## 10. Modelling improvements

- [x] Add --grid-search option for repeated cross-validation and extended
parameter grids.

- [x] port grid search helper for decision tree

- [x] save best estimator when performing cart grid search
- [x] extend cart grid search with min_samples_split and class_weight

- [x] Verify that each function from ai_arisha.py is represented or
intentionally omitted in the src modules (see FUNCTIONS.md).

- [x] port reporting helpers for final artifact collection

- [x] add evaluation_utils helpers for plotting and fairness aliases
- [x] extend safe_transform tests for type error and warning handling

- [x] add tests for calibrate_model isotonic option and invalid method handling

- [x] add CLI test for sampler option

- [x] extend FeatureEngineer unit tests for column normalisation, asset ratios
and risk flag

- [x] add Makefile test target to run pytest
- [x] port `_vif_prune` as `vif_prune` in `src/selection.py` with unit tests
- [x] VIF pruning handles singular matrices
- [x] add random_forest model pipeline with CLI and tests

## 11. Metrics helpers

- [x] Port notebook metrics helpers `eval_metrics`, `eval_at`, `show_metrics`
and `folds_df` or confirm omission.
- [x] Create `src/metrics.py` with unit tests for these functions.
- [x] implement random_split and time_split in src/split.py; add set_seeds
helper in new src/utils.py
- [x] Simplify vif_prune to drop one column per iteration and recalc VIFs
- [x] add `is_binary_numeric` helper in `src/utils.py` with unit tests

- [x] Port sha256, shasum, save_folds and run_grid helpers into src.manifest
with unit tests.

- [x] Ported build_outer_iter and nested_cv with bootstrap fallback into new
src/cv_utils.py with tests.

## 12. Preprocessing validation

- [x] Integrate `validate_prep` into training scripts to fail fast on bad
scaling.

- [x] centralise grid-search helpers as pipeline_helpers

## 13. Utility helpers

- The original notebook defines small helper functions `_zeros`, `_dedup` and
  `_is_binary_numeric`. These create zero-filled series, merge lists without
  duplicates and detect 0/1 numeric columns. They now live in `src/utils.py`
  with unit tests.

- [x] Port these helpers into `src/utils.py` with accompanying unit tests.
- [x] Fix README stray code block marker leaving rest in code.

## 14. Reporting helpers

- [x] Add prefix helper and metrics helpers in report_helpers.py with tests.
- [x] Record in NOTES that `_sha`/`sha` map to `sha256`/`shasum` and
  `_is_binary`, `_num_block` and `make_preprocessor` remain unported.
- [x] Port `_is_binary`, `_num_block` and `make_preprocessor` into
  `src/preprocessing.py` with unit tests.

## 15. Prediction CLI

- [x] Add `src/predict.py` and console script `mlcls-predict` with tests.
- [x] Expand docs with examples on using the prediction command.

## 16. CI docs build

- [x] Build Sphinx HTML in CI and upload as artifact.

- [x] Upgrade docs upload step to `actions/upload-artifact@v4`.

- [x] Update docs job to use upload-artifact@v4.

- [x] Add additional notebook examples demonstrating advanced use cases.
- [x] Update README badges and add `.markdown-link-check.json` to ignore
      LinkedIn links.
- [x] Add Sphinx API reference page listing key modules
- [x] Document fairness checks and calibration in advanced_usage.rst

## 17. Reporting CLI

- [x] expose mlcls-report console script for collecting report artifacts
- [x] document its usage and the ``report_artifacts/`` directory in the docs

## 18. Packaging

- [x] add build-system entry and document wheel creation with `python -m build`
- [x] publish the wheel to PyPI once the release workflow is battle-tested
- [x] add isort hook to pre-commit for import ordering

## 19. Binder support

- [x] add binder/environment.yml referencing requirements
- [x] create postBuild script to install package in editable mode
- [x] update docs if binder instructions change

## 20. Data caching

- [x] check `.sha256` file in `download_data.py` to skip re-downloads

## 20. Fairness metrics

- [x] compute equal opportunity ratio in evaluate.py and document usage

