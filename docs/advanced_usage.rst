Advanced workflows
==================

This page highlights optional steps available when training models. The
``advanced_demo.ipynb`` notebook demonstrates the commands.

Grid search
-----------

Run ``mlcls-train -g`` to search a wider range of parameters. The job records
metrics for each candidate and stores the best estimator under ``artefacts/``.
To perform grid search on the random-forest pipeline pass the model option::

   mlcls-train --model random_forest -g
   mlcls-train --model gboost -g

Calibration
-----------

After training a model you can calibrate the predicted probabilities::

   mlcls-eval --calibrate isotonic

This fits a calibration model on the validation fold and reports Brier score in
the output table. The calibrated estimator is saved with the suffix
``_calibrated.joblib``.

Run the standalone helper to draw reliability plots for both models::

   python -m src.calibration

The script reads the saved models and generates ``*_calibration.png``
files under ``artefacts/``.

Fairness checks
---------------

Use the evaluation command with ``--group-col`` to compute group metrics such
as statistical parity and equal opportunity. The summary table now includes an
``equal_opp`` column showing the worst to best true positive rate ratio::

   mlcls-eval --group-col gender --group-col marital
This command prints parity ratios for each group and stores them in
``artefacts/group_metrics.csv``. ``summary_metrics.csv`` records the
``equal_opp`` ratio for each model. It also stores ``eq_odds`` which is the
difference between the true- and false-positive rate gaps.

Set a custom probability cutoff for these metrics with ``--threshold``. When
omitted the tool chooses the Youden J statistic::

   mlcls-eval --group-col gender --threshold 0.6

Select specific pipelines with ``--models``. Pass multiple names to evaluate
only those models::

   mlcls-eval --models logreg random_forest svm

The ``advanced_demo.ipynb`` notebook walks through these steps and shows the
additional plots.

SHAP values
-----------

Provide a DataFrame to ``logreg_coefficients`` or ``tree_feature_importances``
and pass ``shap_csv_path`` to store per-feature SHAP values::

   from src.feature_importance import logreg_coefficients

   shap_df = logreg_coefficients(
       "artefacts/lr.joblib",
       shap_csv_path="artefacts/logreg_shap_values.csv",
       X=X_test,
   )

The helper function ``compute_shap_values`` creates the table with columns
matching the input DataFrame.

SHAP plots
----------

Use ``plot_shap_summary`` to visualise these values::

   from src.feature_importance import plot_shap_summary

   plot_shap_summary(
       "artefacts/lr.joblib",
       X=X_test,
       png_path="artefacts/logreg_shap.png",
   )

The image ``logreg_shap.png`` appears under ``artefacts/``.

Report artefacts
----------------

Gather recent metrics and plots with the report command::

   mlcls-report

The tool copies the latest files into ``report_artifacts/``. You can zip the
folder and share it with collaborators.
