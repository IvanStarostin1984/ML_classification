# Migration notes

Current commit: `8520315`.

At this point the repo consists only of the original Colab export `ai_arisha.py`,
a project `README.md`, `AGENTS.md` and a data licence notice under `data/`.
None of the planned `src/` modules or CI files exist yet.

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

2025-06-10: Added train/eval modules for logistic regression and decision tree with stratified split utility. Updated Makefile, Dockerfile, README and added basic model tests.
