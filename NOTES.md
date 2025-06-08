# Migration notes

Current commit: `354c4fc`.


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
The next step is to break this large script into smaller modules as outlined in
`TODO.md` and introduce tests plus GitHub Actions.


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

