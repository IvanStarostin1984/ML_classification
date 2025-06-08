# Contributor Guidelines for Stock-App1

This repository hosts ML classifier (trees and logistic classifier).  Follow these rules when adding or updating code.

Your job is to migrate Google Colab notebook ai_arisha.py to several classes github repo in line with README.md to showcase my work.

# Important — Not authoritative:
1. This file is only a quick-start contributor guide.
2. Your job is to adapt google colab python code to Github.
3. KAGGLE_USERNAME, KAGGLE_KEY are in secrets

# Proposed project structure:
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
│   └─ models/
│       ├─ __init__.py
│       ├─ logreg.py                # LR training / eval pipeline
│       └─ cart.py                  # Decision-Tree pipeline
├─ tests/
│   └─ test_smoke.py                # CI sanity import check
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
- Use 2‑space indentation, single quotes and end files with a newline.
- Document each public API/function with a doc comment.


## Testing & CI
- Run tests before committing.

## Contributing Workflow
- **Fork** then branch off `main` using the pattern `feat/<topic>`.
- **Ensure local tests pass** before opening a PR.
- **Each PR requires at least one reviewer.**

to understand the current stage, past decisions, and open questions tied to the spec.

Refer to `README.md` and full documentation in `docs/` for further details on features and architecture.
