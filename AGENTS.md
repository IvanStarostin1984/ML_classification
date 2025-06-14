# Contributor Guidelines for ML_classification

This repository hosts ML classifier (trees and logistic classifier). Follow
these rules when adding or updating code.

Your job is to migrate Google Colab notebook ai_arisha.py to several classes
github repo in line with README.md to showcase my work.

The style rules apply to new code under `src/` and tests, while `ai_arisha.py`
is kept as-is for reference

TODO.md, NOTES.md AGENTS.md may be not perfectly aligned with current project
status. Always recheck in actual code.
**Distinct-files rule**:

1. Every concurrent task must confine its edits to a **unique list of code or
data files**.
2. _Shared exceptions_: any task may append (never rewrite) the markdown logs
`NOTES.md`, `TODO.md`, and this `AGENTS.md`.
3. If two or more open PRs would touch the same non-markdown file, cancel or
re-scope one of them before continuing.

## Important — Not authoritative

1. This file is only a quick-start contributor guide.
2. Your job is to adapt google colab python code to Github.
3. KAGGLE_USERNAME, KAGGLE_KEY are in secrets
4. Never commit downloaded Kaggle data or personal API keys
5. With _ANY_ pull request add data to NOTES.md to shortly reflect on work done
in this pull request. Always!

## Project structure

ML_classification/
├─ .github/
│ └─ workflows/
│ └─ ci.yml # lint + pytest on Python 3.10
├─ ai_arisha.py # original Colab / legacy script
├─ AGENTS.md # design / architecture notes
├─ data/
│ └─ README.md # Kaggle‐licence notice & instructions
├─ notebooks/
│ └─ README.md # slim Colab/Binder demo stub
├─ scripts/
│ └─ download_data.py # pulls dataset via Kaggle API
├─ src/
│ ├─ `__init__.py`
│ ├─ calibration.py # probability calibration utilities
│ ├─ cv_utils.py # cross-validation wrappers
│ ├─ dataprep.py # load / clean raw data
│ ├─ diagnostics.py # χ² tests, corr heat-map, etc.
│ ├─ diagnostics_stats.py # additional stats for diagnostics
│ ├─ evaluate.py # nested CV + fairness metrics
│ ├─ evaluation_utils.py # evaluation helpers
│ ├─ fairness.py # fairness helpers
│ ├─ feature_importance.py # tree-based feature importance
│ ├─ features.py # FeatureEngineer class
│ ├─ manifest.py # dataset manifest utilities
│ ├─ metrics.py # metric utilities from notebook
│ ├─ pipeline_helpers.py # CLI and pipeline orchestrators
│ ├─ preprocessing.py # ColumnTransformers
│ ├─ reporting.py # reporting helpers, CLI `mlcls-report`
│ ├─ report_helpers.py # confusion matrix and group metrics
│ ├─ selection.py # VIF, RFE, tree selector
│ ├─ split.py # stratified train/test logic
│ ├─ train.py # orchestrates pipelines
│ ├─ predict.py # prediction CLI
│ ├─ utils.py # small misc helpers
│ └─ models/
│    ├─ `__init__.py`
│    ├─ logreg.py # LR training / eval pipeline
│    └─ cart.py # Decision-Tree pipeline
├─ tests/
│ ├─ test_calibration.py # probability calibration
│ ├─ test_cart_gridsearch.py # CART grid-search pipeline
│ ├─ test_cli_sampler.py # CLI data sampler functions
│ ├─ test_cli_scripts.py # CLI wrappers
│ ├─ test_cli_train_gridsearch.py # CLI grid search training
│ ├─ test_predict.py # prediction CLI
│ ├─ test_cv_utils.py # cross-validation helpers
│ ├─ test_dataprep.py # data loading
│ ├─ test_diagnostics.py # diagnostic utilities
│ ├─ test_diagnostics_stats.py # extended diagnostics
│ ├─ test_download_data.py # dataset download script
│ ├─ test_evaluate.py # evaluation CLI
│ ├─ test_evaluate_extended.py # extended metrics
│ ├─ test_evaluation_utils.py # evaluation helpers
│ ├─ test_fairness.py # fairness metrics
│ ├─ test_feature_importance.py # feature importance calculators
│ ├─ test_features.py # feature engineering
│ ├─ test_logreg_gridsearch.py # logistic grid-search
│ ├─ test_manifest.py # dataset manifest
│ ├─ test_manifest_plots.py # manifest plotting
│ ├─ test_metrics.py # metric utilities
│ ├─ test_models.py # modelling pipelines
│ ├─ test_oversampling.py # oversampling heuristics
│ ├─ test_pipeline_helpers.py # pipeline helpers
│ ├─ test_preprocessing.py # preprocessing pipeline
│ ├─ test_reporting.py # reporting utilities
│ ├─ test_selection.py # feature selection
│ ├─ test_smoke.py # CI sanity import check
│ ├─ test_split.py # split logic
│ ├─ test_utils.py # miscellaneous utils
├─ environment.yml # Conda spec (Python ≥ 3.10)
├─ requirements.txt # pip fallback
├─ Dockerfile # reproducible container build
├─ Makefile # one-command workflow (`make train`)
├─ .gitignore # excludes data, artefacts, secrets
├─ .markdown-link-check.json # patterns for link checker to ignore
├─ LICENSE # MIT License
└─ README.md # badges, quick-start, results, contact

## Coding Standards

- One domain concept per file; no cyclic imports.
- Functions should stay under 20 lines with at most 2 nesting levels.
- Favour composition over inheritance and keep variables scoped tightly.
- Validate inputs early and throw on bad data.
- Use 4‑space indentation, single quotes and end files with a newline.
- Document each public API/function with a doc comment.

### Markdown style

- Keep lines under 80 characters.
- Surround lists, headings and fenced code blocks with blank lines.
- Use backticks around file names containing underscores to avoid MD050.
- Specify a language for fenced code blocks.
- NOTES.md and TODO.md entries must keep lines under 80 chars and avoid
  multiple blank lines. Run `npx markdownlint-cli` after updates.
- Ensure exactly one blank line between entries in NOTES.md and TODO.md to
  prevent MD012.
- Run `npx markdownlint-cli` locally before committing documentation changes.
- Run `npx markdownlint-cli '**/*.md' --ignore node_modules` to mirror the CI
  job.
- `make lint-docs` runs this command automatically.
- End each Markdown file with exactly one newline to satisfy MD047.

## Testing & CI

- The GitHub Actions workflow (`.github/workflows/ci.yml`) runs `flake8`,
  `black --check .` and `pytest` on Python&nbsp;3.10.
- Run these commands locally before committing to ensure your code passes the
  same checks. Use `make test` to run the full pytest suite with the correct
`PYTHONPATH`.
- Black uses a line length of **88** as configured in `pyproject.toml`.
- Docs-only commits run a fast job with `markdownlint`.
- After tests pass, CI builds the Sphinx docs and uploads them using
  `actions/upload-artifact@v4`. Keep the major version current when GitHub
  releases a new one.
  Links are checked using:

```bash
find . -name '*.md' -not -path '*node_modules*' -print0 |
  xargs -0 -n1 npx markdown-link-check -q
```

The tool reads `.markdown-link-check.json` for patterns of external links to
ignore. Links to social profiles like LinkedIn may be skipped there.

## Contributing Workflow

- **Fork** then branch off `main` using the pattern `feat/<topic>`.
- **Ensure local tests pass** before opening a PR.
- **Each PR requires at least one reviewer.**
- with _every commit_ reflect in **NOTES.md** on work done in a short lean way
to track work.
- keep docs in sync with code; update AGENTS.md and README when behaviour
  changes.
- the project documentation lives under `docs/`; keep it in sync with README and
  code changes.
- install Sphinx from the requirements files before building the docs.
- Sphinx now builds an API reference from docstrings in
  `docs/api_reference.rst`.
- Build distribution packages with `python -m build` to create a wheel.

Read `NOTES.md` and `TODO.md` to understand the current stage, past decisions,
and open questions tied to the spec.

Refer to `README.md` and full documentation in `docs/` for further details on
features and architecture.

## Migration tracking

Two markdown files at the repository root help coordinate the refactor.

- **TODO.md** – complete list of tasks required to move the legacy
`ai_arisha.py` notebook into the modular project layout above.
- **NOTES.md** – running log that explains what was done and how.

Contributors must keep `TODO.md` up to date with remaining work and record
completed items in `NOTES.md`.

- **FUNCTIONS.md** – reference list of notebook functions. Ensure each is
implemented or intentionally skipped. Log skipped ones in NOTES.md.
