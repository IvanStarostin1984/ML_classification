# Functions in ai_arisha.py

This list captures every function defined in the legacy Colab script. Use it
when verifying that the modular `src/` package implements equivalent behaviour.

- `_zeros()` – return Series of zeros aligned to `df_fe` index.
- `_fmt_p(p, thr=1e-6)` – format p-value; shows `<thr` for tiny values.
- `_annotate(ax)` – annotate matplotlib bars with integer heights.
- `_need_exact(exp)` – decide whether chi² test needs Monte Carlo.
- `_cramers_v(chi2, tbl)` – Cramer’s V effect size for contingency tables.
- `_cochran_armitage(ct)` – Cochran–Armitage trend test for 2×k tables.
- `_safe_chi2(ct, need_mc, rng)` – chi² test with Monte Carlo and 0.5
adjustment fallback.
- `_dedup(old, new)` – merge pairs and drop duplicates regardless of order.
- `_is_binary_numeric(series)` – check if numeric column only contains 0/1.
- `safe_transform(preprocessor, X_new, copy=True)` – apply fitted
ColumnTransformer dropping unseen columns.
- `_vif_prune(cols, cap)` – iteratively drop columns with highest VIF until
below cap.
- `_random_split(test_frac)` – stratified random train/test split.
- `_time_split(test_frac)` – time-based train/test split using `DATE_COL`.
- `set_seeds(s=SEED)` – seed Python, NumPy and PYTHONHASHSEED.
- `lr_steps(samp)` – pipeline steps order for logistic regression.
- `tree_steps(samp)` – pipeline steps order for decision tree.
- `run_gs(name, steps, estimator, grid)` – run GridSearchCV and report best
ROC-AUC.
- `show_metrics(lbl, y_true, y_prob, y_hat)` – print test-set metrics summary.
- `_is_binary(series)` – helper to filter out pure binary numeric features.
- `_num_block(cols, scaler)` – return (pipeline, columns) for numeric scaler
block.
- `make_preprocessor(include_cont)` – build ColumnTransformer for LR or tree.
- `_prefix(label)` – extract prefix from feature name.
- `_scaled_matrix(prep)` – transform full dataset and return matrix with
feature names.
- `_check_mu_sigma(mat, idx, tol_mu=1e-3, tol_sd=1e-2)` – verify scaled
columns have mean≈0 and sd≈1.
- `validate_prep(prep, name, check_scale=True)` – run NaN/scale checks on
transformer.
- `_conf(label, yhat)` – print confusion matrix and metrics for threshold.
- `_group_metrics(col)` – compute group-wise TPR for fairness audit.
- `build_outer_iter(y)` – return outer CV splits or bootstrap when
minority<10.
- `nested_cv(pipe, grid, label)` – run nested cross-validation and report
scores.
- `folds_df(res, mdl)` – helper to turn CV results into a tidy DataFrame.
- `youden_thr(est, X, y)` – compute optimal Youden J threshold from ROC curve.
- `four_fifths(est, X, y, col, thr)` – inner function used to compute 4/5ths
ratio.
- `shasum(fp)` – short SHA-256 digest of file bytes.
- `sha256(path)` – full SHA-256 digest of file bytes.
- `save_folds(tag, cv)` – save fold indices produced by CV iterator.
- `run_grid(cv, tag)` – grid-search helper used during LR tuning.
- `_sha(fp)` – short SHA-256 digest; defined in multiple cells.
- `sha(fp)` – same as above but using pathlib.Path.
- `eval_at(th)` – compute test metrics at given probability threshold.
- `eval_metrics(y_true, y_prob, y_pred, suffix)` – return dict of performance
metrics.
- `plot_or_load(fname, plot_fn)` – reuse saved plot if available else
generate.
- `main()` – collect various artifacts into `report_artifacts/`.
  - `find_path(name)` – search artifact directories for file.
  - `read_latest_glob(pattern, dirs=(ART, PLOTS, ROOT))` – return most recent
match.
  - `write_section(f, title, reader_fn)` – helper for report table sections.
  - `flatten_cv(path)` – collapse multi-index CV CSV to two-row table.
  - `flatten_metrics(md)` – flatten nested metric JSON.
  - `dump_dataset_overview(ff)` – write dataset overview text.
  - `dump_lr_params(ff)` – report best logistic regression parameters.
  - `dump_cart_params(ff)` – report best CART parameters.
  - `dump_feature_counts(ff)` – report engineered feature counts from
registry.
  - `dump_corr_top10(ff)` – show top absolute correlations.
  - `dump_dropped_twins(ff)` – list numeric twins removed due to high
  correlation.
  - `dump_skew_profile(ff)` – summarize skewness profile of numeric features.
  - `dump_cart_overfit(ff)` – compare CART CV vs test ROC AUC.

Implemented later:
- `prefix` – now `src.utils.prefix`.
- `conf_matrix_summary` and `group_metrics` – moved to `src.report_helpers`.
