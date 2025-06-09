# Migration notes

Current commit: `8af97fc`.
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
2025-06-08: Marked tasks complete in TODO and expanded migration notes for commit 536978c.
2025-06-08: Rewrote TODO introduction to reference dbd5184 and note that modular code, tests and CI now exist.
2025-06-13: Updated AGENTS.md to list all tests and clarify 4-space indentation standard.
2025-06-14: Marked README update as complete in TODO.
2025-06-14: Updated README to remove missing-modules note and confirm the migration is complete.
2025-06-14: Updated README to remove missing-modules note and confirm the migration is complete.
2025-06-08: Removed obsolete README statements about missing model pipelines and noted make train runs both models.
2025-06-14: Renamed project structure heading in AGENTS.md.
2025-06-08: Refreshed README repository layout and checked TODO item.
2025-06-08: Adjusted CI workflow to invoke pytest via python -m.
2025-06-15: Added PYTHONPATH env var in CI to fix ModuleNotFoundError during tests.
2025-06-16: Added evaluate.py with nested CV and fairness metrics plus tests.
2025-06-16: Added make eval target and expanded README with evaluation instructions and fairness guidance.
2025-06-08: integrated FeatureEngineer into model pipelines and updated tests.
2025-06-08: Added CLI main entry in evaluate.py and updated tests and README.
2025-06-08: Added project metadata in pyproject.toml and exposed src as installable package. README now documents 'pip install -e .' for development.
2025-06-08: Added src/train.py CLI orchestrating both models and updated Makefile to use it.
2025-06-08: Added unit tests for fairness metrics.
2025-06-08: Wrapped VIF computation in warnings and numpy error state contexts to avoid RuntimeWarning when columns are perfectly collinear.
2025-06-08: Clarified Kaggle credential setup in README.
2025-06-17: Updated TODO intro to mark migration complete and added evaluation/fairness checklist item.
2025-06-08: Documented evaluate/train/fairness modules and tests in AGENTS.md directory tree.
2025-06-18: Cleaned tests and formatted selection/train modules.
2025-06-08: Expanded Testing & CI guidelines in AGENTS.md to describe flake8, black, pytest workflow.
2025-06-18: documented new mlcls-* console scripts and usage.
2025-06-08: Updated download_data to check CSV existence before using Kaggle API and expanded tests.
2025-06-08: Added console script entrypoints and tests invoking them.
2025-06-08: refactored FeatureEngineer.transform into helper methods to meet function length rule. Added docstrings and updated tests. Black and pytest fail due to pyproject parsing error.
2025-06-18: Added --data-path option to mlcls-train and updated tests.
2025-06-20: Added CITATION.cff for citation metadata.
2025-06-21: Added plotting helpers and manifest writer with tests.
2025-06-08: expanded evaluate metrics and CV, added new tests
2025-06-21: Added calibration module with CLI and tests for model probability calibration.
2025-06-08: added feature_importance module exporting logistic coefficients and tree importances with tests.
2025-06-21: Added sampler option to training pipeline and oversampling tests.
2025-06-22: Documented sampler CLI, calibration command, feature-importance outputs and manifest in README.
2025-06-22: Cleaned TODO to remove outdated missing-feature notes.
2025-06-08: Removed extra blank lines in src/__init__.py to satisfy flake8.
2025-06-08: Fixed indentation in train-cart, train, eval commands in Makefile.
2025-06-09: Verified Kaggle download and training pipelines. Added lowercase loan_status handling in dataprep.
2025-06-09: Strip whitespace in dataset columns for evaluation.
2025-06-23: Replaced Build & Test badge with GitHub internal badge for private repo.
2025-06-23: download_data warns when src package is missing and tests cover it.
2025-06-23: Added note in README that 'pip install -e .' registers src for import so scripts like python scripts/download_data.py work.
2025-06-24: README clarifies that `make` is required and lists console script alternatives for Windows.
2025-06-09: Added grid_train_from_df using GridSearchCV with repeated CV and unit test for parameter grid.
2025-06-09: Added grid_train_from_df with grid search and tests.
2025-06-25: Documented --grid-search option for exhaustive cross-validation and added TODO bullet.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-09: Added grid-search flag to mlcls-train and tests.
2025-06-25: Cleaned logreg.grid_train_from_df to use RepeatedStratifiedKFold and removed duplicate docstring.
2025-06-09: Fixed stray parameter block in cart.grid_train_from_df; function now returns fitted GridSearchCV.
2025-06-24: Marked TODO item clarifying Makefile usage as done.
2025-06-09: Documented grid-search flag and dataset size in README.
2025-06-09: Verified grid_train_from_df header and removed stray blank line.
2025-06-09: mlcls-train now prints best cart grid-search score.

2025-06-09: Added FUNCTIONS.md summarising all functions from ai_arisha.py for verification.

<details><summary>Codex log</summary>

```
        1  ls
        2  cd ML_classification
        3  ls
        4  cat AGENTS.md
        5  cat TODO.md
        6  cat NOTES.md
        7  cat pyproject.toml
        8  ls
        9  wc -l ai_arisha.py
       10  grep -n "^def" -n ai_arisha.py | head
       11  grep -n "^def" -n ai_arisha.py | tail
       12  grep -n "def " ai_arisha.py | head -n 20
       13  grep -n "def " ai_arisha.py | sed -n '21,40p'
       14  grep -n "def " ai_arisha.py | sed -n '41,80p'
       15  grep -n "def " ai_arisha.py | sed -n '81,120p'
       16  grep -n "def " ai_arisha.py | wc -l
       17  grep -n "def " ai_arisha.py > /tmp/defs.txt
       18  cat /tmp/defs.txt
       19  grep -n "def " ai_arisha.py | sed -n '50,90p'
       20  sed -n '120,170p' ai_arisha.py
       21  sed -n '140,150p' ai_arisha.py
       22  find . -name AGENTS.md -print
       23  head README.md
       24  grep -n "FUNCTIONS.md" -r .
       25  tail -n 20 NOTES.md
       26  tail -n 20 TODO.md
       27  cat .github/workflows/ci.yml
       28  ls .github/workflows
       29  flake8 && echo OK
       30  pip install flake8 black pytest pandas numpy scikit-learn matplotlib seaborn imbalanced-learn scipy statsmodels joblib kaggle
       31  flake8 && echo OK
       32  black --check . && echo OK
       33  pytest -q && echo OK
       34  sed -n '140,155p' ai_arisha.py
       35  sed -n '372,406p' ai_arisha.py
       36  sed -n '404,450p' ai_arisha.py
       37  grep -n "def _dedup" -n ai_arisha.py
       38  sed -n '712,750p' ai_arisha.py
       39  grep -n "_is_binary_numeric" -n ai_arisha.py
       40  sed -n '760,770p' ai_arisha.py
       41  grep -n "safe_transform" -n ai_arisha.py
       42  sed -n '852,892p' ai_arisha.py
       43  sed -n '892,910p' ai_arisha.py
       44  sed -n '1000,1032p' ai_arisha.py
       45  grep -n "_random_split" -n ai_arisha.py
       46  sed -n '1176,1196p' ai_arisha.py
       47  grep -n "set_seeds" -n ai_arisha.py
       48  sed -n '1330,1345p' ai_arisha.py
       49  sed -n '1384,1436p' ai_arisha.py
       50  sed -n '1432,1490p' ai_arisha.py
       51  grep -n "_is_binary(" -n ai_arisha.py
       52  sed -n '1528,1615p' ai_arisha.py
       53  sed -n '1612,1660p' ai_arisha.py
       54  sed -n '1660,1680p' ai_arisha.py
       55  sed -n '1680,1705p' ai_arisha.py
       56  grep -n "_conf(" -n ai_arisha.py
       57  sed -n '1896,1925p' ai_arisha.py
       58  sed -n '1925,1947p' ai_arisha.py
       59  sed -n '1947,1985p' ai_arisha.py
       60  grep -n "build_outer_iter" -n ai_arisha.py
       61  sed -n '2000,2060p' ai_arisha.py
       62  sed -n '2060,2090p' ai_arisha.py
       63  grep -n "shasum" -n ai_arisha.py
       64  sed -n '2090,2168p' ai_arisha.py
       65  sed -n '2168,2195p' ai_arisha.py
       66  sed -n '2196,2260p' ai_arisha.py
       67  sed -n '2260,2330p' ai_arisha.py
       68  grep -n "_sha(fp):" -n ai_arisha.py
       69  sed -n '2570,2592p' ai_arisha.py
       70  sed -n '2592,2618p' ai_arisha.py
       71  sed -n '2720,2740p' ai_arisha.py
       72  sed -n '2740,2760p' ai_arisha.py
       73  grep -n "four_fifths" -n ai_arisha.py
       74  sed -n '3360,3395p' ai_arisha.py
       75  sed -n '3395,3425p' ai_arisha.py
       76  sed -n '3425,3450p' ai_arisha.py
       77  grep -n "eval_at" -n ai_arisha.py
       78  sed -n '3108,3140p' ai_arisha.py
       79  sed -n '3140,3210p' ai_arisha.py
       80  grep -n "eval_metrics(" -n ai_arisha.py
       81  sed -n '3248,3280p' ai_arisha.py
       82  grep -n "plot_or_load" -n ai_arisha.py
       83  sed -n '3328,3336p' ai_arisha.py
       84  sed -n '3336,3368p' ai_arisha.py
       85  grep -n "def main" -n ai_arisha.py
       86  sed -n '3528,3590p' ai_arisha.py
       87  sed -n '3590,3660p' ai_arisha.py
       88  sed -n '3660,3700p' ai_arisha.py
       89  sed -n '3700,3744p' ai_arisha.py
       90  sed -n '3744,3778p' ai_arisha.py
       91  sed -n '3778,3814p' ai_arisha.py
       92  sed -n '3814,3844p' ai_arisha.py
       93  cat /tmp/defs.txt
       94  cat > FUNCTIONS.md <<'EOF'
    # Functions in ai_arisha.py
    
    This list captures every function defined in the legacy Colab script. Use it when verifying that the modular `src/` package implements equivalent behaviour.
    
    - `_zeros()` – return Series of zeros aligned to `df_fe` index.
    - `_fmt_p(p, thr=1e-6)` – format p-value; shows `<thr` for tiny values.
    - `_annotate(ax)` – annotate matplotlib bars with integer heights.
    - `_need_exact(exp)` – decide whether chi² test needs Monte Carlo.
    - `_cramers_v(chi2, tbl)` – Cramer’s V effect size for contingency tables.
    - `_cochran_armitage(ct)` – Cochran–Armitage trend test for 2×k tables.
    - `_safe_chi2(ct, need_mc, rng)` – chi² test with Monte Carlo and 0.5 adjustment fallback.
    - `_dedup(old, new)` – merge pairs and drop duplicates regardless of order.
    - `_is_binary_numeric(series)` – check if numeric column only contains 0/1.
    - `safe_transform(preprocessor, X_new, copy=True)` – apply fitted ColumnTransformer dropping unseen columns.
    - `_vif_prune(cols, cap)` – iteratively drop columns with highest VIF until below cap.
    - `_random_split(test_frac)` – stratified random train/test split.
    - `_time_split(test_frac)` – time-based train/test split using `DATE_COL`.
    - `set_seeds(s=SEED)` – seed Python, NumPy and PYTHONHASHSEED.
    - `lr_steps(samp)` – pipeline steps order for logistic regression.
    - `tree_steps(samp)` – pipeline steps order for decision tree.
    - `run_gs(name, steps, estimator, grid)` – run GridSearchCV and report best ROC-AUC.
    - `show_metrics(lbl, y_true, y_prob, y_hat)` – print test-set metrics summary.
    - `_is_binary(series)` – helper to filter out pure binary numeric features.
    - `_num_block(cols, scaler)` – return (pipeline, columns) for numeric scaler block.
    - `make_preprocessor(include_cont)` – build ColumnTransformer for LR or tree.
    - `_prefix(label)` – extract prefix from feature name.
    - `_scaled_matrix(prep)` – transform full dataset and return matrix with feature names.
    - `_check_mu_sigma(mat, idx, tol_mu=1e-3, tol_sd=1e-2)` – verify scaled columns have mean≈0 and sd≈1.
    - `validate_prep(prep, name, check_scale=True)` – run NaN/scale checks on transformer.
    - `_conf(label, yhat)` – print confusion matrix and metrics for threshold.
    - `_group_metrics(col)` – compute group-wise TPR for fairness audit.
    - `build_outer_iter(y)` – return outer CV splits or bootstrap when minority<10.
    - `nested_cv(pipe, grid, label)` – run nested cross-validation and report scores.
    - `folds_df(res, mdl)` – helper to turn CV results into a tidy DataFrame.
    - `youden_thr(est, X, y)` – compute optimal Youden J threshold from ROC curve.
    - `four_fifths(est, X, y, col, thr)` – inner function used to compute 4/5ths ratio.
    - `shasum(fp)` – short SHA-256 digest of file bytes.
    - `sha256(path)` – full SHA-256 digest of file bytes.
    - `save_folds(tag, cv)` – save fold indices produced by CV iterator.
    - `run_grid(cv, tag)` – grid-search helper used during LR tuning.
    - `_sha(fp)` – short SHA-256 digest; defined in multiple cells.
    - `sha(fp)` – same as above but using pathlib.Path.
    - `eval_at(th)` – compute test metrics at given probability threshold.
    - `eval_metrics(y_true, y_prob, y_pred, suffix)` – return dict of performance metrics.
    - `plot_or_load(fname, plot_fn)` – reuse saved plot if available else generate.
    - `main()` – collect various artifacts into `report_artifacts/`.
      - `find_path(name)` – search artifact directories for file.
      - `read_latest_glob(pattern, dirs=(ART, PLOTS, ROOT))` – return most recent match.
      - `write_section(f, title, reader_fn)` – helper for report table sections.
      - `flatten_cv(path)` – collapse multi-index CV CSV to two-row table.
      - `flatten_metrics(md)` – flatten nested metric JSON.
      - `dump_dataset_overview(ff)` – write dataset overview text.
      - `dump_lr_params(ff)` – report best logistic regression parameters.
      - `dump_cart_params(ff)` – report best CART parameters.
      - `dump_feature_counts(ff)` – report engineered feature counts from registry.
      - `dump_corr_top10(ff)` – show top absolute correlations.
      - `dump_dropped_twins(ff)` – list numeric twins removed due to high correlation.
      - `dump_skew_profile(ff)` – summarize skewness profile of numeric features.
      - `dump_cart_overfit(ff)` – compare CART CV vs test ROC AUC.
    EOF
    
       95  cat FUNCTIONS.md
       96  sed -n '1,5p' FUNCTIONS.md
       97  echo >> TODO.md
       98  tail -n 5 TODO.md
       99  printf "- [ ] Verify that each function from ai_arisha.py is represented or intentionally omitted in the src modules (see FUNCTIONS.md).\n" >> TODO.md
      100  printf '%s
    ' '- [ ] Verify that each function from ai_arisha.py is represented or intentionally omitted in the src modules (see FUNCTIONS.md).' >> TODO.md
      101  tail -n 8 TODO.md
      102  date
      103  printf '\n2025-06-09: Added FUNCTIONS.md summarising all functions from ai_arisha.py for verification.\n' >> NOTES.md
      104  tail -n 1 NOTES.md
      105  grep -n "Codex" -n NOTES.md
      106  head -n 5 NOTES.md
      107  sed -n '1,40p' NOTES.md
      108  history | tail -n 20
      109  history | head -n 1
      110  history
      111  tail -n 2 NOTES.md
      112  truncate -s -1 NOTES.md
      113  tail -n 2 NOTES.md
      114  sed -i '$d' NOTES.md
      115  tail -n 1 NOTES.md
      116  printf '2025-06-09: Added FUNCTIONS.md summarising all functions from ai_arisha.py for verification.\n<details><summary>Codex log</summary>\n\n```
    ' >> NOTES.md
      117  history | sed 's/^/    /' >> NOTES.md
      118  printf '\n```\n</details>\n' >> NOTES.md
      119  tail -n 15 NOTES.md
      120  grep -n "Codex log" -n NOTES.md
      121  sed -n '100,115p' NOTES.md
      122  sed -n '115,150p' NOTES.md
      123  sed -n '150,185p' NOTES.md
      124  sed -n '185,220p' NOTES.md
      125  sed -n '220,255p' NOTES.md
      126  sed -n '255,290p' NOTES.md
      127  tail -n 5 NOTES.md
      128  tail -n 10 NOTES.md
      129  nl -ba NOTES.md | sed -n '100,130p'
      130  nl -ba NOTES.md | sed -n '130,170p'
      131  nl -ba NOTES.md | sed -n '170,210p'
      132  nl -ba NOTES.md | sed -n '210,250p'
      133  nl -ba NOTES.md | sed -n '250,290p'
      134  tail -n 5 NOTES.md
      135  grep -n "Added FUNCTIONS.md" -n NOTES.md
      136  nl -ba NOTES.md | sed -n '100,110p'
      137  grep -n '</details>' -n NOTES.md
      138  sed -n '102,289p' NOTES.md | head
      139  sed -n '289,$p' NOTES.md
      140  sed -n '100,130p' NOTES.md
      141  wc -l NOTES.md
      142  head -n 102 NOTES.md > /tmp/NOTES_prefix
      143  tail -n +289 NOTES.md > /tmp/NOTES_suffix
      144  cat /tmp/NOTES_prefix /tmp/NOTES_suffix > NOTES.md
      145  tail -n 5 NOTES.md
      146  printf '\n<details><summary>Codex log</summary>\n\n```\n' >> NOTES.md
      147  history | sed 's/^/    /' >> NOTES.md

```
</details>
