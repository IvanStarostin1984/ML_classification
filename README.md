# ML_classification

> **A tidy, production-ready re-implementation of my Google Colab notebook for
> predicting loan approvals with logistic regression, decision-tree and
> random-forest pipelines.**

[![Build & Test][badge-ci]][ci-link]
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![ROC-AUC 0.987 ± 0.008][roc-badge]]

---

## What’s inside & why it matters

* **End-to-end pipeline** – data download, cleaning, 80 + engineered features,
  rigorous feature selection, model tuning, and statistical evaluation.
* **Statistical transparency** – every performance number is reported with
  95 % bootstrap confidence intervals and a fairness check (four-fifths rule).
* **Clean architecture** – each stage is its own Python module under `src/`,
  ready for unit tests and continuous integration.
* **One-command reproducibility** – `make train` or run the Docker image from
  the provided `Dockerfile` to train the models and regenerate all artefacts.
* **CI/CD ready** – GitHub Actions lint + pytest on every push.

* **Modular utilities** – feature engineering and diagnostics are available as
  importable helpers. Helpers like `split.random_split`, `split.time_split` and
  `utils.set_seeds` simplify experiments.

The rendered documentation lives at
<https://ivanstarostin1984.github.io/ML_classification>.

See `CHANGELOG.md` for release notes.

---

## Quick-start

```bash
# Clone the repo
git clone https://github.com/IvanStarostin1984/ML_classification.git
cd ML_classification

# Set up the environment
pip install -r requirements.txt
# or: conda env create -f environment.yml

# Install the project in editable mode for development
pip install -e .

# Enable automatic formatting on commits
pip install pre-commit
pre-commit install

# Running pre-commit needs network access or a `GIT_TOKEN` with
# at least the `public_repo` scope. Store the token as a secret and
# reference it in CI.

# The hooks run `isort` before `black` and `flake8` so imports stay ordered.
# In CI the workflow runs `pre-commit run --files` on changed files before
# `flake8`, `black` and `pytest`.

# This registers the `src` package so scripts like
# `python scripts/download_data.py` can import it.

# If you used conda, activate the environment
conda activate ml-classification

# Provide your Kaggle API token before downloading the dataset.
# Either place `kaggle.json` under `~/.kaggle/` or export the
# `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables:
#
#   export KAGGLE_USERNAME=your_username
#   export KAGGLE_KEY=your_key
#
# Download the Kaggle dataset
python scripts/download_data.py

# The raw CSVs land in `data/raw/` (git-ignored).
# A `.sha256` file keeps the checksum so the script skips re-downloading
# if the dataset hasn't changed.

# Train, evaluate and store artefacts in artefacts/
make train            # run both models
make eval             # evaluate trained models and check fairness
# or individually
make train-logreg
make train-cart
mlcls-train --model random_forest  # train only the RF model
mlcls-train --model random_forest -g  # grid search for the RF model
mlcls-train --model gboost  # train the gradient boosting model
mlcls-train --model gboost -g  # grid search for gradient boosting
mlcls-train --model svm  # train the support vector machine
mlcls-train --model svm -g  # grid search for SVM
mlcls-train --sampler smote   # run with SMOTE oversampling
```

Pre-commit hooks format code and lint Markdown automatically on each commit.
They run `isort`, `black` and `flake8` when you commit.

Create a personal access token with the `public_repo` permission at
<https://github.com/settings/tokens> and store it as `GIT_TOKEN` under your
repository secrets. CI exports this token so `pre-commit` can clone its hook
repositories without prompts.

Note: `make` is required for these commands. On Windows, install GNU Make or run
the console scripts `mlcls-train` and `mlcls-eval` instead.

See [data/README.md](data/README.md) for dataset licence notes.

Interactive notebooks live under `notebooks/`. Open `loan_demo.ipynb` or
`advanced_demo.ipynb` for a guided walkthrough.
You can also launch them instantly on Binder via the badge in
`notebooks/README.md`.
Binder sessions do not ship with the Kaggle dataset and cannot download
it without credentials.

Training produces feature-importance tables (`logreg_coefficients.csv`,
`cart_importances.csv`) and bar-chart PNGs in `artefacts/`. All generated files
are recorded in `artefacts/SHA256_manifest.txt` for reproducibility.
Pass a DataFrame to `logreg_coefficients` or `tree_feature_importances`
along with `shap_csv_path` to save SHAP value tables as well.
Use `plot_shap_summary` to turn those values into a PNG stored in
`artefacts/`.

`make eval` runs `python -m src.evaluate` to compute test metrics and the worst
 four-fifths ratio across protected groups (pass `--group-col` to override the
default). Metrics are stored in `artefacts/summary_metrics.csv` and printed to
stdout. A ratio below **0.8** warns of possible bias.
You can replicate the notebook's exhaustive cross-validation using the training
command with `--grid-search` (or `-g`):

```bash
mlcls-train --grid-search  # repeated CV with extended parameter grids
```

This run takes longer but mirrors the notebook results.

For fairness evaluation and calibration instructions see
[docs/advanced_usage.rst](docs/advanced_usage.rst).

## Running tests

Execute the test-suite locally with:

```bash
make test
```

This sets `PYTHONPATH` so `pytest` can find the `src` package.

### Local testing

Install the requirements before running the tests:

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

## Building the docs

Install Sphinx from `requirements.txt` or `environment.yml` first:

```bash
pip install -r requirements.txt  # or conda env update -f environment.yml
```

Then generate HTML pages:

```bash
make docs         # or cd docs && sphinx-build -b html . _build
```

The output appears under `docs/_build/`.

Use `make lint-docs` to check Markdown files.

A personal access token with the `contents:write` scope must be stored in
the `GH_PAGES_TOKEN` repository secret so the docs workflow can push to the
`gh-pages` branch. Without this secret the `gh-pages` job fails with
"not found deploy key or tokens".

## Building a wheel

Install the build tool and run:

```bash
python -m pip install build
python -m build
```

The wheel lands in `dist/`.

Tagged releases run the same build in CI and attach the wheel to a GitHub
release. Tag a commit with `git tag v1.2.3` and push it to trigger the upload
to PyPI via `twine`.

## Command-line usage

After installing the project in editable mode you get these console commands:

```bash
pip install -e .
mlcls-train          # trains both models
mlcls-train --model random_forest -g  # extensive grid search
mlcls-train --model gboost -g  # gradient boosting grid search
mlcls-train --model svm -g  # SVM grid search
mlcls-eval --threshold 0.6  # sets fairness metric cutoff
mlcls-predict        # generates predictions from a saved model
mlcls-report        # collects report artifacts
mlcls-manifest      # writes checksums for selected files
mlcls-summary       # prints dataset statistics
```

Example usage:

```bash
mlcls-summary --data-path data/raw/loan_approval_dataset.csv
```

These commands require the Kaggle dataset, which is distributed under its
original licence. See [data/README.md](data/README.md) for details. The dataset
is small – around 380&nbsp;kB (~1000 rows) – so the default training run
finishes in a few seconds. Pass `-g` to `mlcls-train` to perform the extensive
grid search (5×3 cross-validation) used in the original notebook.
See `docs/cli_usage.rst` for a walkthrough of these commands.

**Prefer Docker?**

```bash
docker build -t ml_classification .
docker run --rm \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY ml_classification
```

## Model calibration

Run the calibration helper after training to create reliability plots:

```bash
python -m src.calibration
```

This saves `logreg_calibration.png` and `cart_calibration.png` (plus
`*_calibrated.joblib` models) in `artefacts/`.

---

## Repository layout

The project follows the target directory layout. Running `make train` now
executes both the logistic regression and decision-tree pipelines located under
`src/models`.

```text
ai_arisha.py             ← legacy Colab script (read-only)
AGENTS.md                ← contributor guidelines and architecture notes
.github/workflows/ci.yml ← CI pipeline (Black, flake8, pytest)
scripts/download_data.py ← Kaggle dataset pull helper
src/                     ← Python package skeleton
src/models/logreg.py     ← logistic regression pipeline
src/models/cart.py       ← decision-tree pipeline
src/models/random_forest.py ← random-forest pipeline
src/models/gradient_boosting.py ← gradient boosting pipeline
src/features.py          ← FeatureEngineer class
src/diagnostics.py       ← chi-square & correlation plots
src/preprocessing.py     ← ColumnTransformer helpers
src/selection.py         ← VIF & tree-based selector
src/calibration.py       ← probability calibration CLI
src/evaluation_utils.py  ← evaluation helpers
src/cv_utils.py          ← cross-validation utilities
src/manifest.py          ← SHA-256 manifest writer
src/feature_importance.py← importance tables
src/pipeline_helpers.py  ← grid-search utilities
src/reporting.py         ← report assembly helpers
src/diagnostics_stats.py ← stats for diagnostics
src/utils.py             ← general helpers
tests/                   ← pytest suite
data/README.md           ← dataset licence notes
notebooks/README.md      ← Colab/Binder demo stub
binder/environment.yml   ← Binder spec
binder/postBuild         ← install step
Dockerfile, Makefile     ← reproducible build & workflow helpers
environment.yml          ← Conda spec (Python ≥ 3.10)
pyproject.toml           ← project build metadata
requirements.txt         ← pip fallback
LICENSE                  ← MIT
README.md                ← you are here
```

---

## Key results (hold-out test set)

| Model               |  ROC-AUC  | PR-AUC | Fairness (4/5 rule)    |
| ------------------- | :-------: | :----: | ---------------------- |
| Logistic Regression | **0.987** |  0.991 | Pass (gender, marital) |
| Decision Tree       |   0.961   |  0.972 | Pass (gender, marital) |

Values reproduced from the accompanying statistical report.&#x20;

---

## How to cite

```bibtex
@misc{Starostin2025LoanApproval,
  author = {Ivan Starostin},
  title  = {ML\_classification: Loan-approval prediction pipelines},
  year   = {2025},
  url    = {https://github.com/IvanStarostin1984/ML_classification}
}
```

See [CITATION.cff](CITATION.cff) for other citation formats.

---

## Author

**Ivan Starostin** – [LinkedIn](https://www.linkedin.com/in/ivanstarostin/)

[roc-badge]: https://img.shields.io/static/v1?label=Test%20ROC-AUC&message=0.987%C2%B10.008&color=purple
[badge-ci]: https://img.shields.io/github/actions/workflow/status/IvanStarostin1984/ML_classification/ci.yml?branch=main
[ci-link]: https://github.com/IvanStarostin1984/ML_classification/actions
