# ML_classification

> **A tidy, production-ready re-implementation of my Google Colab notebook for predicting loan approvals with logistic regression and decision-tree pipelines.**

[![Build & Test](https://github.com/IvanStarostin1984/ML_classification/actions/workflows/ci.yml/badge.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![ROC-AUC 0.987 ± 0.008](https://img.shields.io/badge/Test ROC–AUC-0.987-±0.008-purple)

---

## What’s inside & why it matters

* **End-to-end pipeline** – data download, cleaning, 80 + engineered features, rigorous feature selection, model tuning, and statistical evaluation.
* **Statistical transparency** – every performance number is reported with 95 % bootstrap confidence intervals and a fairness check (four-fifths rule).
* **Clean architecture** – each stage is its own Python module under `src/`, ready for unit tests and continuous integration.
* **One-command reproducibility** – `make train` or `docker compose up` trains the models and regenerates all artefacts.
* **CI/CD ready** – GitHub Actions lint + pytest on every push.

* **Modular utilities** – feature engineering and diagnostics are available as importable helpers.
---

## Quick-start

```bash
# Clone the repo
git clone https://github.com/IvanStarostin1984/ML_classification.git
cd ML_classification

# Set up the environment
pip install -r requirements.txt          # or: conda env create -f environment.yml

# Install the project in editable mode for development
pip install -e .

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

# Train, evaluate and store artefacts in artefacts/
make train            # run both models
make eval             # evaluate trained models and check fairness
# or individually
make train-logreg
make train-cart
mlcls-train --sampler smote   # run with SMOTE oversampling
```

Note: `make` is required for these commands. On Windows, install GNU Make or run
the console scripts `mlcls-train` and `mlcls-eval` instead.

See [data/README.md](data/README.md) for dataset licence notes.

Training produces feature-importance tables (`logreg_coefficients.csv`,
`cart_importances.csv`) and bar-chart PNGs in `artefacts/`. All generated files
are recorded in `artefacts/SHA256_manifest.txt` for reproducibility.

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


## Running tests

Execute the test-suite locally with:

```bash
make test
```
This sets `PYTHONPATH` so `pytest` can find the `src` package.


## Command-line usage

After installing the project in editable mode you get two console commands:

```bash
pip install -e .
mlcls-train          # trains both models
mlcls-train -g       # extensive grid search
mlcls-eval           # evaluates the trained models
```

These commands require the Kaggle dataset, which is distributed under its
original licence. See [data/README.md](data/README.md) for details. The dataset
is small – around 380&nbsp;kB (~1000 rows) – so the default training run
finishes in a few seconds. Pass `-g` to `mlcls-train` to perform the extensive
grid search (5×3 cross-validation) used in the original notebook.

**Prefer Docker?**

```bash
docker build -t ml_classification .
docker run --rm -e KAGGLE_USERNAME=$KAGGLE_USERNAME -e KAGGLE_KEY=$KAGGLE_KEY ml_classification
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

```
ai_arisha.py             ← legacy Colab script (read-only)
AGENTS.md                ← contributor guidelines and architecture notes
.github/workflows/ci.yml ← CI pipeline (Black, flake8, pytest)
scripts/download_data.py ← Kaggle dataset pull helper
src/                     ← Python package skeleton
src/models/logreg.py     ← logistic regression pipeline
src/models/cart.py       ← decision-tree pipeline
src/features.py          ← FeatureEngineer class
src/diagnostics.py       ← chi-square & correlation plots
src/preprocessing.py     ← ColumnTransformer helpers
src/selection.py         ← VIF & tree-based selector
tests/                   ← pytest suite
data/README.md           ← dataset licence notes
notebooks/README.md      ← Colab/Binder demo stub
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

```
@misc{Starostin2025LoanApproval,
  author = {Ivan Starostin},
  title  = {ML\_classification: Loan-approval prediction pipelines},
  year   = {2025},
  url    = {https://github.com/IvanStarostin1984/ML_classification}
}
```

---

## Author

**Ivan Starostin** – [LinkedIn](https://www.linkedin.com/in/ivanstarostin/)

