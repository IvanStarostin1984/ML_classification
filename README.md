# ML_classification

> **A tidy, production-ready re-implementation of my Google Colab notebook for predicting loan approvals with logistic regression and decision-tree pipelines.**

[![Build & Test](https://img.shields.io/github/actions/workflow/status/IvanStarostin1984/ML_classification/ci.yml?branch=main)](../../actions)  
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

---

## Quick-start

```bash
# Clone the repo
git clone https://github.com/IvanStarostin1984/ML_classification.git
cd ML_classification

# Set up the environment
pip install -r requirements.txt          # or: conda env create -f environment.yml

# Download the Kaggle dataset (needs KAGGLE_USERNAME and KAGGLE_KEY env vars)
python scripts/download_data.py

# Train, evaluate and store artefacts in artefacts/
make train
```

**Prefer Docker?**

```bash
docker build -t ml_classification .
docker run --rm -e KAGGLE_USERNAME=$KAGGLE_USERNAME -e KAGGLE_KEY=$KAGGLE_KEY ml_classification
```

---

## Repository layout

```
ai_arisha.py            ← original Colab notebook (read-only legacy)
AGENTS.md               ← contributor guidelines & architecture notes
.github/workflows/ci.yml← CI pipeline (Black, flake8, pytest)
scripts/download_data.py← Kaggle dataset pull helper
src/                    ← modular pipeline (dataprep, features, models…)
tests/                  ← pytest suite
Dockerfile, Makefile    ← reproducible build & one-command workflow
environment.yml         ← conda spec (Python ≥ 3.10)
requirements.txt        ← pip fallback
LICENSE                 ← MIT
README.md               ← you are here
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

