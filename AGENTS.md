# Contributor Guidelines for ML_classification

This repository hosts ML classifier (trees and logistic classifier).  Follow these rules when adding or updating code.

Your job is to migrate Google Colab notebook ai_arisha.py to several classes github repo in line with README.md to showcase my work.

The style rules apply to new code under `src/` and tests, while `ai_arisha.py` is kept as-is for reference

TODO.md, NOTES.md AGENTS.md may be not perfectly aligned with current project status. Always recheck in actual code.
**Distinct-files rule**:  
   1. Every concurrent task must confine its edits to a **unique list of code or data files**.  
   2. *Shared exceptions*: any task may append (never rewrite) the markdown logs `NOTES.md`, `TODO.md`, and this `AGENTS.md`.  
   3. If two or more open PRs would touch the same non-markdown file, cancel or re-scope one of them before continuing.

# Important — Not authoritative:
1. This file is only a quick-start contributor guide.
2. Your job is to adapt google colab python code to Github.
3. KAGGLE_USERNAME, KAGGLE_KEY are in secrets
4. Never commit downloaded Kaggle data or personal API keys
5. With *ANY* pull request add data to NOTES.md to shortly reflect on work done in this pull request. Always! 

# Project structure
ML_classification/
├─ .github/
│   └─ workflows/
│       └─ ci.yml                   # lint + pytest on Python 3.10
├─ ai_arisha.py                     # original Colab / legacy script
├─ AGENTS.md                        # design / architecture notes
├─ data/
│   └─ README.md                    # Kaggle‐licence notice & instructions
├─ notebooks/
│   └─ README.md                    # slim Colab/Binder demo stub
├─ scripts/
│   └─ download_data.py             # pulls dataset via Kaggle API
├─ src/
│   ├─ __init__.py
│   ├─ dataprep.py                  # load / clean raw data
│   ├─ features.py                  # FeatureEngineer class
│   ├─ diagnostics.py               # χ² tests, corr heat-map, etc.
│   ├─ preprocessing.py             # ColumnTransformers
│   ├─ selection.py                 # VIF, RFE, tree selector
│   ├─ split.py                     # stratified train/test logic
│   ├─ evaluate.py                  # nested CV + fairness metrics
│   ├─ fairness.py                  # fairness helpers
│   ├─ train.py                     # orchestrates pipelines
│   └─ models/
│       ├─ __init__.py
│       ├─ logreg.py                # LR training / eval pipeline
│       └─ cart.py                  # Decision-Tree pipeline
├─ tests/
│   ├─ test_dataprep.py             # unit tests for data loading
│   ├─ test_features.py             # unit tests for feature engineering
│   ├─ test_models.py               # unit tests for modelling pipelines
│   ├─ test_smoke.py                # CI sanity import check
│   ├─ test_download_data.py        # tests the data download script
│   ├─ test_preprocessing.py        # preprocessing pipeline tests
│   ├─ test_selection.py            # feature selection helpers
│   ├─ test_diagnostics.py          # diagnostic utilities
│   ├─ test_evaluate.py             # tests for the evaluation CLI
│   └─ test_fairness.py             # tests for fairness metrics
├─ environment.yml                  # Conda spec (Python ≥ 3.10)
├─ requirements.txt                 # pip fallback
├─ Dockerfile                       # reproducible container build
├─ Makefile                         # one-command workflow (`make train`)
├─ .gitignore                       # excludes data, artefacts, secrets
├─ LICENSE                          # MIT License
└─ README.md                        # badges, quick-start, results, contact



## Coding Standards
- One domain concept per file; no cyclic imports.
- Functions should stay under 20 lines with at most 2 nesting levels.
- Favour composition over inheritance and keep variables scoped tightly.
- Validate inputs early and throw on bad data.
- Use 4‑space indentation, single quotes and end files with a newline.
- Document each public API/function with a doc comment.


## Testing & CI
- Run tests before committing.

## Contributing Workflow
- **Fork** then branch off `main` using the pattern `feat/<topic>`.
- **Ensure local tests pass** before opening a PR.
- **Each PR requires at least one reviewer.**
- with *every commit* reflect in **NOTES.md** on work done in a short lean way to track work.

Read `NOTES.md` and `TODO.md` to understand the current stage, past decisions, and open questions tied to the spec.

Refer to `README.md` and full documentation in `docs/` for further details on features and architecture.

## Migration tracking

Two markdown files at the repository root help coordinate the refactor.

- **TODO.md** – complete list of tasks required to move the legacy `ai_arisha.py` notebook into the modular project layout above.
- **NOTES.md** – running log that explains what was done and how.

Contributors must keep `TODO.md` up to date with remaining work and record completed items in `NOTES.md`.
