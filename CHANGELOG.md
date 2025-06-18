# Changelog

## 0.1.0 - 2025-09-05

- Initial release with modular pipelines for logistic regression, decision tree,
  random forest and gradient boosting.
- Command-line scripts for data download, training, evaluation and reporting.
- Fairness metrics implementing the four-fifths rule.
- SHAP visualisation helper for feature importance.
- Binder environment, Dockerfile and Makefile for reproducible runs.
- CI workflow running linting and tests plus Sphinx documentation.

## 0.1.1 - 2025-09-08

- Documented that pre-commit requires network access or a GIT_TOKEN.
- Linked CITATION.cff from README and docs.
- Added equalized_odds_diff metric with eq_odds column.

## 0.1.2 - 2025-09-26

- Added `mlcls-summary` CLI for quick dataset statistics.
- Replaced secret checks with helper steps to avoid YAML errors.
- Enforced trailing space rule via markdownlint.

## 0.1.3 - 2025-10-02

- Documented `mlcls-summary` command in README.
- Public dataset summary CLI available through `mlcls-summary`.
