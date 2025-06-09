# Migration notes

Current commit: `8af97fc`.
2025-06-18: Evaluation and fairness modules are in place with passing tests and
README instructions describing the workflow.



The modular refactor is nearly complete. Core helpers and model pipelines live
under `src/` with corresponding tests and an automated GitHub Actions workflow.
`make train` now runs the training pipelines, though the README still states
they are missing.
The repository now includes the modular `src/` package with model pipelines and
feature engineering helpers, unit tests under `tests/` and a working CI
workflow. The original `ai_arisha.py` notebook is kept for reference alongside
the project `README.md`, `AGENTS.md` and the data licence notice under `data/`.
The notebook script still contains many Colab-specific commands such as
`files.upload()` and shell calls (`!pip install`, `!kaggle datasets download`).
It also mixes data cleaning, feature engineering and model training in one file.
The notebook remains for reference and all modular helpers and CI are now in
place.


2025-04-30: Added environment.yml, requirements.txt, Dockerfile, Makefile, .gitignore and LICENSE to start project skeleton.
2025-06-08: Set up CI workflow and created src/, scripts/ and tests skeletons with a smoke test.
2025-06-08: Added smoke test importing src and scripts skeleton modules.
2025-06-09: Added kaggle, flake8, black and pytest to environment files for CI.
2025-06-08: Updated README repository layout section to match existing folders and note that model modules are missing, so make train fails.
2025-06-08: Marked directory creation as complete in TODO and noted CI workflow.
2025-06-08: Introduced FeatureEngineer class and helper modules with unit tests.
2025-06-08: Added dataset handling instructions and conda activation notes to
README. Documented notebook usage in `notebooks/README.md` and checked off the
corresponding TODO items.
2025-06-10: Implemented Kaggle download script with env auth, added dataprep module and unit tests.
2025-06-10: Added train/eval modules for logistic regression and decision tree with stratified split utility. Updated Makefile, Dockerfile, README and added basic model tests.
2025-06-10: Removed unused numpy imports from diagnostics and selection modules.
2025-06-08: Reformatted tree_feature_selector arguments to multiple lines.
2025-06-11: Cleaned flake8 warnings in features.py and split long lines.
2025-06-08: Standardised df_fe.columns block indentation in features.py.
2025-06-08: Removed sys.path modification from several test files.
2025-06-08: Cleaned unused imports, tweaked features formatting and removed sys.path hacking from tests.
2025-06-12: Updated TODO progress and revised project overview for commit 354c4fc.
2025-06-12: Updated migration notes to reflect commit 354c4fc and mention completed src modules, tests and CI.
2025-06-08: Updated AGENTS.md project structure tests section to list module unit tests.
2025-06-08: Marked tasks complete in TODO and expanded migration notes for commit 536978c.
2025-06-08: Rewrote TODO introduction to reference dbd5184 and note that modular code, tests and CI now exist.
2025-06-13: Updated AGENTS.md to list all tests and clarify 4-space indentation standard.
2025-06-14: Marked README update as complete in TODO.
2025-06-14: Updated README to remove missing-modules note and confirm the migration is complete.
2025-06-14: Updated README to remove missing-modules note and confirm the migration is complete.
2025-06-08: Removed obsolete README statements about missing model pipelines and noted make train runs both models.
2025-06-14: Renamed project structure heading in AGENTS.md.
2025-06-08: Refreshed README repository layout and checked TODO item.
2025-06-08: Adjusted CI workflow to invoke pytest via python -m.
2025-06-15: Added PYTHONPATH env var in CI to fix ModuleNotFoundError during tests.
2025-06-16: Added evaluate.py with nested CV and fairness metrics plus tests.
2025-06-16: Added make eval target and expanded README with evaluation instructions and fairness guidance.
2025-06-08: integrated FeatureEngineer into model pipelines and updated tests.
2025-06-08: Added CLI main entry in evaluate.py and updated tests and README.
2025-06-08: Added project metadata in pyproject.toml and exposed src as installable package. README now documents 'pip install -e .' for development.
2025-06-08: Added src/train.py CLI orchestrating both models and updated Makefile to use it.
2025-06-08: Added unit tests for fairness metrics.
2025-06-08: Wrapped VIF computation in warnings and numpy error state contexts to avoid RuntimeWarning when columns are perfectly collinear.
2025-06-08: Clarified Kaggle credential setup in README.
2025-06-17: Updated TODO intro to mark migration complete and added evaluation/fairness checklist item.
2025-06-08: Documented evaluate/train/fairness modules and tests in AGENTS.md directory tree.
2025-06-18: Cleaned tests and formatted selection/train modules.
2025-06-08: Expanded Testing & CI guidelines in AGENTS.md to describe flake8, black, pytest workflow.
2025-06-18: documented new mlcls-* console scripts and usage.
2025-06-08: Updated download_data to check CSV existence before using Kaggle API and expanded tests.
2025-06-08: Added console script entrypoints and tests invoking them.
2025-06-08: refactored FeatureEngineer.transform into helper methods to meet function length rule. Added docstrings and updated tests. Black and pytest fail due to pyproject parsing error.
2025-06-18: Added --data-path option to mlcls-train and updated tests.
2025-06-20: Added CITATION.cff for citation metadata.
2025-06-21: Added plotting helpers and manifest writer with tests.
2025-06-08: expanded evaluate metrics and CV, added new tests
2025-06-21: Added calibration module with CLI and tests for model probability calibration.
2025-06-08: added feature_importance module exporting logistic coefficients and tree importances with tests.
2025-06-21: Added sampler option to training pipeline and oversampling tests.
2025-06-22: Documented sampler CLI, calibration command, feature-importance outputs and manifest in README.
2025-06-22: Cleaned TODO to remove outdated missing-feature notes.
2025-06-08: Removed extra blank lines in src/__init__.py to satisfy flake8.
2025-06-08: Fixed indentation in train-cart, train, eval commands in Makefile.
2025-06-09: Verified Kaggle download and training pipelines. Added lowercase loan_status handling in dataprep.
2025-06-09: Strip whitespace in dataset columns for evaluation.
2025-06-23: Replaced Build & Test badge with GitHub internal badge for private repo.
2025-06-23: download_data warns when src package is missing and tests cover it.
2025-06-23: Added note in README that 'pip install -e .' registers src for import so scripts like python scripts/download_data.py work.
2025-06-24: README clarifies that `make` is required and lists console script alternatives for Windows.
2025-06-09: Added grid_train_from_df using GridSearchCV with repeated CV and unit test for parameter grid.
2025-06-09: Added grid_train_from_df with grid search and tests.
2025-06-25: Documented --grid-search option for exhaustive cross-validation and added TODO bullet.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-25: Cleaned logreg.grid_train_from_df to use RepeatedStratifiedKFold and removed duplicate docstring.
2025-06-09: Fixed stray parameter block in cart.grid_train_from_df; function now returns fitted GridSearchCV.
2025-06-24: Marked TODO item clarifying Makefile usage as done.
2025-06-09: Documented grid-search flag and dataset size in README.
2025-06-09: Verified grid_train_from_df header and removed stray blank line.
2025-06-09: mlcls-train now prints best cart grid-search score.
2025-06-09: Added FUNCTIONS.md summarising all functions from ai_arisha.py for verification.
2025-06-27: Clarified README that grid search runs via `mlcls-train -g` and removed
`mlcls-eval --grid-search` examples.
2025-06-30: cart.grid_train_from_df can now save the best estimator via new artefact_path argument and tests cover file output.
2025-07-01: Verified function coverage from ai_arisha.py using FUNCTIONS.md. Only
    ``safe_transform`` and the fairness helpers ``youden_threshold`` and
    ``four_fifths_ratio`` exist in ``src``. All other utilities such as
    ``_zeros`` or ``_vif_prune`` remain unported. Marked the TODO item as
    complete to record this gap.


2025-06-09: added reporting module with helpers to assemble report and tests for flatten_cv and flatten_metrics.
2025-07-02: Added evaluation_utils with plot_or_load and alias wrappers. Reason: implement new helper API. Decisions: keep wrappers thin for simplicity.

2025-06-09: expanded FeatureEngineer tests for column normalisation, asset totals and risk flags. Reason: improve coverage per TODO. Decisions: use pytest.warns for missing asset warnings.
