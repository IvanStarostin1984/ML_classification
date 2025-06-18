# Migration notes

Current commit: `97db366`.

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

2025-04-30: Added environment.yml, requirements.txt, Dockerfile, Makefile,
.gitignore and LICENSE to start project skeleton.
2025-06-08: Set up CI workflow and created src/, scripts/ and tests skeletons
with a smoke test.
2025-06-08: Added smoke test importing src and scripts skeleton modules.
2025-06-09: Added kaggle, flake8, black and pytest to environment files for CI.
2025-06-08: Updated README repository layout section to match existing folders
and note that model modules are missing, so make train fails.
2025-06-08: Marked directory creation as complete in TODO and noted CI workflow.
2025-06-08: Introduced FeatureEngineer class and helper modules with unit tests.
2025-06-08: Added dataset handling instructions and conda activation notes to
README. Documented notebook usage in `notebooks/README.md` and checked off the
corresponding TODO items.
2025-06-10: Implemented Kaggle download script with env auth, added dataprep
module and unit tests.
2025-06-10: Added train/eval modules for logistic regression and decision tree
with stratified split utility. Updated Makefile, Dockerfile, README and added
basic model tests.
2025-06-10: Removed unused numpy imports from diagnostics and selection modules.
2025-06-08: Reformatted tree_feature_selector arguments to multiple lines.
2025-06-11: Cleaned flake8 warnings in features.py and split long lines.
2025-06-08: Standardised df_fe.columns block indentation in features.py.
2025-06-08: Removed sys.path modification from several test files.
2025-06-08: Cleaned unused imports, tweaked features formatting and removed
sys.path hacking from tests.
2025-06-12: Updated TODO progress and revised project overview for commit
354c4fc.
2025-06-12: Updated migration notes to reflect commit 354c4fc and mention
completed src modules, tests and CI.
2025-06-08: Updated AGENTS.md project structure tests section to list module
unit tests.
2025-06-08: Marked tasks complete in TODO and expanded migration notes for
commit 536978c.
2025-06-08: Rewrote TODO introduction to reference dbd5184 and note that
modular code, tests and CI now exist.
2025-06-13: Updated AGENTS.md to list all tests and clarify 4-space indentation
standard.
2025-06-14: Marked README update as complete in TODO.
2025-06-14: Updated README to remove missing-modules note and confirm the
migration is complete.
2025-06-14: Updated README to remove missing-modules note and confirm the
migration is complete.
2025-06-08: Removed obsolete README statements about missing model pipelines
and noted make train runs both models.
2025-06-14: Renamed project structure heading in AGENTS.md.
2025-06-08: Refreshed README repository layout and checked TODO item.
2025-06-08: Adjusted CI workflow to invoke pytest via python -m.
2025-06-15: Added PYTHONPATH env var in CI to fix ModuleNotFoundError during
tests.
2025-06-16: Added evaluate.py with nested CV and fairness metrics plus tests.
2025-06-16: Added make eval target and expanded README with evaluation
instructions and fairness guidance.
2025-06-08: integrated FeatureEngineer into model pipelines and updated tests.
2025-06-08: Added CLI main entry in evaluate.py and updated tests and README.
2025-06-08: Added project metadata in pyproject.toml and exposed src as
installable package. README now documents 'pip install -e .' for development.
2025-06-08: Added src/train.py CLI orchestrating both models and updated
Makefile to use it.
2025-06-08: Added unit tests for fairness metrics.
2025-06-08: Wrapped VIF computation in warnings and numpy error state contexts
to avoid RuntimeWarning when columns are perfectly collinear.
2025-06-08: Clarified Kaggle credential setup in README.
2025-06-17: Updated TODO intro to mark migration complete and added
evaluation/fairness checklist item.
2025-06-08: Documented evaluate/train/fairness modules and tests in AGENTS.md
directory tree.
2025-06-18: Cleaned tests and formatted selection/train modules.
2025-06-08: Expanded Testing & CI guidelines in AGENTS.md to describe flake8,
black, pytest workflow.
2025-06-18: documented new mlcls-\* console scripts and usage.
2025-06-08: Updated download_data to check CSV existence before using Kaggle
API and expanded tests.
2025-06-08: Added console script entrypoints and tests invoking them.
2025-06-08: refactored FeatureEngineer.transform into helper methods to meet
function length rule. Added docstrings and updated tests. Black and pytest fail
due to pyproject parsing error.
2025-06-18: Added --data-path option to mlcls-train and updated tests.
2025-06-20: Added CITATION.cff for citation metadata.
2025-06-21: Added plotting helpers and manifest writer with tests.
2025-06-08: expanded evaluate metrics and CV, added new tests
2025-06-21: Added calibration module with CLI and tests for model probability
calibration.
2025-06-08: added feature_importance module exporting logistic coefficients and
tree importances with tests.
2025-06-21: Added sampler option to training pipeline and oversampling tests.
2025-06-22: Documented sampler CLI, calibration command, feature-importance
outputs and manifest in README.
2025-06-22: Cleaned TODO to remove outdated missing-feature notes.
2025-06-08: Removed extra blank lines in src/**init**.py to satisfy flake8.
2025-06-08: Fixed indentation in train-cart, train, eval commands in Makefile.
2025-06-09: Verified Kaggle download and training pipelines. Added lowercase
loan_status handling in dataprep.
2025-06-09: Strip whitespace in dataset columns for evaluation.
2025-06-23: Replaced Build & Test badge with GitHub internal badge for private
repo.
2025-06-23: download_data warns when src package is missing and tests cover it.
2025-06-23: Added note in README that 'pip install -e .' registers src for
import like
import python
import so
import work.

import scripts
import scripts/download_data.py

2025-06-24: README clarifies that `make` is required and lists console script
alternatives for Windows.
2025-06-09: Added grid_train_from_df using GridSearchCV with repeated CV and
unit test for parameter grid.
2025-06-09: Added grid_train_from_df with grid search and tests.
2025-06-25: Documented --grid-search option for exhaustive cross-validation and
added TODO bullet.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-25: Cleaned logreg.grid_train_from_df to use RepeatedStratifiedKFold
and removed duplicate docstring.
2025-06-09: Fixed stray parameter block in cart.grid_train_from_df; function
now returns fitted GridSearchCV.
2025-06-24: Marked TODO item clarifying Makefile usage as done.
2025-06-09: Documented grid-search flag and dataset size in README.
2025-06-09: Verified grid_train_from_df header and removed stray blank line.
2025-06-09: mlcls-train now prints best cart grid-search score.
2025-06-09: Added FUNCTIONS.md summarising all functions from ai_arisha.py for
verification.
2025-06-27: Clarified README that grid search runs via `mlcls-train -g` and
removed
`mlcls-eval --grid-search` examples.
2025-06-30: cart.grid_train_from_df can now save the best estimator via new
artefact_path argument and tests cover file output.
2025-07-01: Verified function coverage from ai_arisha.py using FUNCTIONS.md.
Besides
`safe_transform` and the fairness helpers `youden_threshold` and
`four_fifths_ratio`, reporting utilities like `find_path` and `write_section`,
along with `flatten_cv` and `flatten_metrics` live in `src/reporting.py`.
All helpers are now ported. `_zeros` lives in `src/utils.py`.
`_vif_prune` moved to `src/selection.py`. Marked the TODO item as complete.
2025-07-02: Updated README layout with new modules list and replaced
docker-compose reference with Dockerfile instructions.
2025-07-02: Tidied TODO numbering and removed duplicate vif_prune item
to keep the task list concise.
2025-06-10: Updated AGENTS.md project structure with all modules and expanded
test list; added docs-sync guideline.

2025-07-02: Fixed README code block closure by replacing the stray "```text```"
line with a closing code fence.

2025-07-03: Documented Markdown guidelines in AGENTS.md. Noted running
`npx markdownlint-cli` before committing docs. Reason: standardise doc style.
Decision: bullet under Coding Standards.

2025-07-02: Fixed README code fence by removing stray `text` line.
2025-07-03: Added Markdown style notes to AGENTS.md and suggested running
`npx markdownlint-cli` after edits.
2025-07-02: Fixed README code block closure by replacing the stray `text`
line with a closing code fence.
2025-06-11: Fixed markdownlint issues across docs and updated README links.
2025-07-04: Added note in AGENTS.md to keep NOTES.md and TODO.md lines under
80 chars and run `npx markdownlint-cli` after edits.

2025-07-02: Fixed README code block closure by replacing the stray "```text```"
line with a closing code fence.

2025-07-03: Documented Markdown style guidelines in AGENTS.md and noted running
`npx markdownlint-cli` before committing docs. Reason: to standardise doc
formatting. Decisions: added bullet list under Coding Standards.

2025-07-02: Fixed README code block closure by replacing the stray
```text``` line with a closing code fence.

2025-07-02: Fixed README code block closure by replacing the stray "```text```"
line with a closing code fence.

2025-06-11: Fixed markdownlint issues across docs and updated README links.

2025-07-16: Shortened long NOTES lines and removed stray blank lines so
markdownlint passes.

2025-07-03: Added TODO item for fixing long lines in NOTES.md to satisfy
markdownlint. Reason: enforce doc style. Decision: bullet under docs updates.

2025-07-17: Added bullet to run markdownlint with glob to match CI checks.

2025-07-20: Removed extra blank lines in NOTES.md to satisfy markdownlint.

2025-07-21: Removed extra blank lines for markdownlint compliance.
2025-07-22: CI link check fixed to iterate over markdown files;
quoting glob failed before.
2025-07-23: Added zeros_like and dedup_pairs in utils with tests. Reason:
  complete TODO for porting notebook helper functions.
2025-07-23: logistic and cart pipelines validate preprocessing before model
training; tests mock `validate_prep` to ensure invocation. Reason: to fail fast
on bad scaling and complete TODO item.
2025-07-24: Added prefix helper and new report_helpers module with
conf_matrix_summary and group_metrics functions plus unit tests.
Reason: port remaining notebook utilities for metrics summarisation.
Decisions: expose via `__all__` and document in FUNCTIONS.md.
2025-07-24: Documented that `_sha` and `sha` were replaced by `sha256` and
`shasum`. `_is_binary`, `_num_block` and `make_preprocessor` have no direct
equivalent. Reason: clarify function coverage and close TODO.
2025-07-24: Clarified that `_zeros` and `_vif_prune` now reside in
`src/utils.py` and `src/selection.py` and updated TODO text.

2025-07-25: Implemented `_is_binary`, `_num_block` and `make_preprocessor`
 in `src/preprocessing.py` with unit tests. Reason: port missing helpers.
 Decisions: simplified make_preprocessor to use a single scaler for
 continuous columns.

2025-07-25: Updated NOTES commit hash to 78a6950 and cleaned trailing
blank lines.
 Reason: keep history accurate.
 Decision: wrap `__all__` in backticks to satisfy markdownlint.

2025-07-25: prefix, conf_matrix_summary and group_metrics implement the
notebook helpers `_prefix`, `_conf` and `_group_metrics`.

2025-07-26: cart.grid_train_from_df tunes min_samples_split and class_weight
 to mirror the notebook search. Tests assert 24 grid combos and verify the
 best estimator is returned. Reason: extend tree grid search in TODO.

2025-06-13: Added `mlcls-predict` CLI to apply saved models. Tests cover the
command and README lists it. Reason: enable simple batch prediction.

2025-07-26: Added minimal Sphinx docs under `docs/` with `conf.py` and
 `index.rst`. Updated README with build instructions and noted docs location
 in AGENTS. Reason: establish documentation framework. Decision: keep default
 Alabaster theme.

2025-07-27: Added CLI usage page with sample commands.
Linked from README and docs index. Marked TODO items as done.

2025-07-28: Added Sphinx to requirements and conda env.
README now tells users to install it before building docs.
AGENTS notes the dependency.

2025-07-29: CI builds Sphinx docs after tests. Makefile has new docs target and
README points to `make docs`.

2025-07-29: Created loan_demo.ipynb demoing data loading, feature
engineering and model training via src package. Updated notebooks/README.md
with link and run steps.

2025-07-30: Upgraded docs upload step to actions/upload-artifact@v4 in CI.
Reason: keep workflow in sync with GitHub action updates.

2025-06-13: Switched docs job to actions/upload-artifact@v4 to keep CI current.

2025-06-14: Added advanced_usage docs summarising grid search, calibration and
fairness referencing advanced_demo.ipynb. Linked from index. Reason: provide
advanced workflow overview.

2025-07-31: Replaced inline ROC-AUC badge with reference link to keep README
lines short. Reason: markdownlint flagged the long badge line.

2025-07-31: Added advanced_demo.ipynb demonstrating grid-search training,
fairness evaluation and calibration. Updated README files accordingly and
checked off the notebook TODO item.

2025-08-01: Updated README badges and added ignore file for LinkedIn links.

2025-08-02: Added API reference page in docs using automodule directives and
inserted it in the index. AGENTS notes that Sphinx builds an API reference.
Reason: document public modules.

2025-08-05: Added build-system with setuptools.build_meta and noted wheel
building via `python -m build` in README and AGENTS. Reason: enable
packaging.

2025-08-03: Expanded advanced_usage.rst with fairness and calibration steps and
linked it from index and README. Reason: clarify advanced evaluation.

2025-06-14: Added mlcls-report CLI with test and docs. CITATION stayed same.

2025-08-06: Documented MD047 newline rule in AGENTS and re-ran markdownlint.
Reason: keep docs lint clean.

2025-08-15: Added trailing newline to NOTES.md to satisfy markdownlint MD047.

2025-08-16: Documented single blank line rule in AGENTS and added link-check
ignore for actions URL. Reason: enforce MD012 and fix link check.

2025-08-16: Removed extra blank line in NOTES to satisfy MD012.

2025-08-16: Logged TODO for markdownlint hook and trimmed blank line.

2025-08-17: Added `lint-docs` target and docs mention.
Completes markdownlint TODO.

2025-08-17: Documented mlcls-report usage in cli_usage.rst and explained the
report_artifacts folder. Reason: show how to collect results for sharing.

2025-08-18: Added release workflow building wheel on tag push and uploading
    as a GitHub release asset. Reason: automate distribution.

2025-08-18: Added pre-commit hooks for black, flake8 and markdownlint.
Reason: enforce consistent formatting automatically.

2025-08-18: Added Binder setup (environment.yml and postBuild) and badge link.
Documented binder folder in README and AGENTS. Reason: enable online notebooks.

2025-08-20: Release workflow now uploads the wheel to PyPI using twine and
README instructs tagging a version to trigger it. Reason: finalise packaging
automation.

2025-08-19: Added isort hook before black and noted it in AGENTS. Reason:
keep imports consistent.

2025-08-26: Documented how to launch notebooks on Binder in notebooks README
and linked the badge from docs/index.rst. Reason: clarify quick start and
complete Binder docs TODO.

2025-08-27: Updated README pre-commit section. Note that isort, black and
    flake8 run automatically. Reason: clarify formatting automation.
    Decision: ticked TODO item.

2025-08-27: Cleaned up packaging roadmap. Checked off isort hook item
because pre-commit already runs isort.

2025-08-28: `download_data.py` now writes a `.sha256` checksum and reuses it to
skip downloading if the CSV is unchanged. Updated README and added
tests for the caching logic.

2025-08-28: Added equal_opportunity_ratio aliasing four_fifths_ratio and
updated evaluate_models to write an ``equal_opp`` column. Updated docs and
tests accordingly.

2025-08-28: CI now runs pre-commit on changed files before flake8, black and
pytest. Updated AGENTS and README. Fixed markdownlint hook pattern in
.pre-commit-config.yaml. Reason: enforce hooks in pipeline.

2025-08-29: Added gh-pages workflow deploying docs and linked the hosted
documentation in README. Updated AGENTS accordingly.

2025-08-29: Added random_forest model mirroring logreg/cart with CLI support,
tests and documentation. Reason: extend modelling options. Decisions: use
RandomForestClassifier with simple grid search and expose via train.py.

2025-08-30: Documented requirement that new code must include unit tests
 and CLI tests in AGENTS.md. Reason: enforce coverage.

2025-08-30: Added RF grid search unit and CLI tests for better coverage.

2025-08-30: Documented random_forest grid search example in docs and README.
Reason: clarify advanced training options.

2025-08-31: Documented that pre-commit downloads hooks from GitHub.
It may prompt for credentials; set GIT_TOKEN to avoid prompts.
Reason: avoid interactive runs when network access is restricted.

2025-08-31: Wrapped long README and NOTES lines under 80 characters and
confirmed with markdownlint. Reason: keep docs compliant with style guide.

2025-08-31: Documented that Markdown-only commits run markdownlint and link
check instead of full tests in AGENTS.md. Reason: clarify CI behaviour.

2025-08-31: Added gradient boosting model and CLI option with grid search
tests. Reason: extend modelling options.
2025-06-14: Added SHAP utilities and optional SHAP export in
feature_importance. Reason: user request to inspect model contributions.

2025-09-01: Documented gradient boosting and SHAP modules in AGENTS.md.

2025-09-02: Documented plot_or_load helper and SHAP PNG size limit in AGENTS.
Mentioned running markdownlint after doc edits.

2025-09-02: Added plot_shap_summary saving SHAP bar plots, integrated optional
shap_png_path into feature_importance and wrote tests for PNG outputs. Reason:
complete TODO for SHAP visualisation.

2025-09-02: Documented new plot_shap_summary helper in README and docs.
Reason: show how to generate SHAP plots.

2025-09-03: Clarified token scopes in AGENTS. PAT with public_repo or repo
is enough for pre-commit. Added note that gh-pages push requires a token with
contents:write. Reason: avoid CI failures on forks.

2025-06-15: Renumbered Fairness metrics to section 21 and Docs hosting to 22
Reason: fix section numbers for clarity.

2025-09-03: Split long doc entry about plot_or_load and markdownlint.
Reason: keep NOTES under 80 characters as per guidelines.

2025-09-05: Added CHANGELOG and noted in AGENTS that releases must update it.
Reason: track feature history for each version.

2025-09-04: Added equalized_odds_diff metric and eq_odds column with tests and
docs. Reason: extend fairness metrics per TODO.

2025-09-04: README and docs index link CITATION.cff so users know how to cite.
Reason: surface citation metadata.
2025-09-07: README states pre-commit needs network access or a GIT_TOKEN.
Token with public_repo scope can be kept as a CI secret. Reason: clarify setup.

2025-06-15: README and docs explain installing requirements for make test.
Reason: clarify local testing.

2025-09-08: Bumped version to 0.1.1 and updated CHANGELOG with token docs,
CITATION link and equalized odds metric.
Decision: emphasise version rule in AGENTS.

2025-09-08: Tagged v0.1.1 release in git to invoke release workflow.
Reason: mark stable package for upload.

2025-09-09: evaluate.py accepts --threshold to override Youden J.
Tests and docs cover it. Reason: allow custom cutoff in fairness metrics.

2025-06-16: Documented Binder limitation that the Kaggle dataset is
absent and cannot be downloaded without credentials. Updated Binder
sections in README, notebooks README, docs index and AGENTS.
Reason: avoid confusion when running demos online.

2025-09-10: Removed TODO item about running a pre-commit hook for markdownlint.
Reason: docs rely on CI gate only.

2025-06-16: Added mlcls-manifest CLI for checksum manifests. Docs and tests
updated. Reason: expose artifact verification.

2025-09-11: Added example in advanced_usage showing mlcls-report saving
artifacts under report_artifacts/ and noted zipping for sharing. Updated
index with reference and ticked TODO. Reason: clarify reporting workflow.

2025-09-12: README shows mlcls-eval --threshold for fairness cutoff.
Reason: document custom cutoff for group metrics.

2025-06-16: Model pipelines import CSV_PATH and reuse it as DATA_PATH.
Reason: centralise default data path.

2025-06-16: Added SVM model with grid search, CLI option and tests. Updated docs
and AGENTS. Reason: expand modelling options per TODO.

2025-06-16: Model pipelines import CSV_PATH and reuse it as DATA_PATH.
Reason: centralise default data path.

2025-09-13: Added mlcls-summary CLI for dataset stats (rows, cols, balance).
Reason: implement TODO item for quick overview.
Decisions: compute stats on cleaned data.
2025-09-14: gh-pages workflow now uses GH_PAGES_TOKEN personal token and job
  gated on repository name. AGENTS updated documenting secret.

2025-09-14: ci.yml passes GIT_TOKEN to pre-commit. Updated AGENTS.
  Reason: ensure hooks clone without prompts.

2025-09-14: Updated isort hook in pre-commit config to use
  `args: ['--profile', 'black']` as required by isort 6.
  Attempted to run `pre-commit` but cloning hook repos failed due to
  missing GitHub token.

2025-09-15: Documented troubleshooting tip for pre-commit failing with
  'could not read Username' in AGENTS.
  Reason: help diagnose GIT_TOKEN errors. Decision: advise checking
  network and secret.

2025-09-16: Marked GH_PAGES_TOKEN secret task as done in TODO.
Reason: docs deployment uses this token.

2025-09-14: gh-pages workflow now uses GH_PAGES_TOKEN secret and checks the
 repository name. ci.yml passes GIT_TOKEN to pre-commit so hooks clone
without prompts. AGENTS updated.

2025-09-17: gh-pages workflow only deploys when GH_PAGES_TOKEN is set.
Reason: avoid failing docs job on forks lacking the secret.

2025-09-18: gh-pages workflow wraps GH_PAGES_TOKEN check in ${{ }} and AGENTS
now warns to do this for all secrets. Reason: avoid YAML parser errors.

2025-06-16: Noted GH_PAGES_TOKEN requirement in README under docs section.
Reason: clarify needed token to deploy pages.

2025-09-19: pre-commit step now checks GIT_TOKEN exists in ci.yml.
Maintainers must define the secret for full checks. Updated AGENTS.

2025-09-20: Quoted pre-commit GIT_TOKEN check in ci.yml to avoid YAML errors.
2025-09-20: AGENTS says run actionlint and quote secrets to avoid YAML issues.

2025-06-16: Removed merge markers in NOTES and wrapped lines. Reason: tidy docs.

2025-09-20: Quote GIT_TOKEN condition in ci.yml to avoid YAML errors.
Reason: follow AGENTS rule on secrets in workflow.

2025-09-20: Added rule to run actionlint on workflow edits.
Secret conditions must be quoted in AGENTS.
Reason: keep workflows linted and avoid YAML issues.

2025-09-20: Added actionlint pre-commit hook and CI step to lint workflows.
Reason: catch YAML mistakes automatically.

2025-09-20: pre-commit token check in ci.yml is quoted to avoid YAML parser
errors. Reason: ensures expression parsing works on all runners.

2025-09-21: Verified gh-pages workflow uses quoted GH_PAGES_TOKEN check and ran
actionlint.

2025-06-17: ensured GIT_TOKEN if quote style and ran actionlint.

2025-09-22: Documented actionlint reminder and exact secret check syntax in
AGENTS to prevent "Unrecognized named-value: 'secrets'" errors.

2025-09-23: Workflows now use helper steps to detect secrets.
Quoting secrets in if conditions is invalid, so this avoids YAML errors.
It clarifies CI logic.

2025-09-23: Added note that quoting secrets in `if:` caused that error and now
use a helper step to check secrets.

2025-09-23: Marked obsolete quoting task and added helper-step detection TODO.

2025-09-24: Clarified secret checks in AGENTS.
Removed outdated quoting guidance and kept
helper-step approach as the only method.
Updated TODO accordingly.

2025-09-24: Previous notes about quoting secrets are obsolete.
All workflows now use a helper step to detect secrets.

2025-09-23: Ticked TODO items for secret helper-step pattern. Completed tasks
for checking secrets via helper steps and removing secret checks from `if:`
conditions to avoid YAML errors.
2025-09-25: Cleaned trailing spaces in NOTES and updated AGENTS.
Added rule about removing them.

2025-09-26: Bumped version to 0.1.2 documenting mlcls-summary CLI,
secret helper steps and trailing space rule. Reason: prepare new
release with helper-step guidance.

2025-09-26: Removed duplicate mlcls-eval line from README. Kept single
example under Command-line usage with mlcls-summary.

2025-09-27: CI now pins actionlint step to v1.7.7 to match pre-commit.
Reason: avoid unexpected lint changes from newer versions.

2025-09-27: Documented pinning rhysd/actionlint to a patch tag in AGENTS to
avoid breakage.

2025-09-28: Documented how missing GIT_TOKEN or blocked network can cause
pre-commit 'failed to authenticate to GitHub' errors. Check
~/.cache/pre-commit/pre-commit.log for details.

2025-06-17: evaluate_models cleans data with dataprep.clean at function start.
Test updated to work with numeric labels. Reason: normalise Loan_Status early.

2025-09-29: Removed conflict markers from NOTES and TODO. Added cleanup item in TODO.md.

2025-09-30: flake8 uses indent-size 4. Updated AGENTS.
Reason: enforce 4-space indentation.

2025-09-30: dataset_summary import exposed in src package `__init__` so
`from src import dataset_summary` works. Added unit test verifying the
function is callable. Reason: align public API with docs.

2025-06-18: added blank lines around TODO headings.
Marked dataset_summary item complete.
Reason: keep formatting consistent and reflect completion.

2025-09-30: `dataset_summary` import exposed in src package `__init__` so
`from src import dataset_summary` works. Added unit test verifying the
function is callable. Reason: align public API with docs.
2025-06-18: Fixed NOTES formatting for markdownlint. Merged conflict marker.
Wrapped `dataset_summary` and `__init__`.
2025-10-01: AGENTS Markdown bullet now mentions wrapping module names in
backticks like `__init__`. Reason: avoid MD050.

2025-06-18: Version bumped to 0.1.3 with README example for `mlcls-summary`.
Reason: document dataset summary CLI before tagging release.

2025-10-02: Documented mlcls-summary usage in CLI docs
and added src.summary to the API reference.
Reason: user request for dataset summary documentation.

2025-10-05: README quick-start clarifies that CI requires the GIT_TOKEN secret.
It explains how to create a PAT and store it as that secret.
Reason: user request for clearer setup.
