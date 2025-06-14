Advanced workflows
==================

This page highlights optional steps available when training models. The
``advanced_demo.ipynb`` notebook demonstrates the commands.

Grid search
-----------

Run ``mlcls-train -g`` to search a wider range of parameters. The job records
metrics for each candidate and stores the best estimator under
``artefacts/``.

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
as statistical parity and equal opportunity::

   mlcls-eval --group-col gender --group-col marital
This command prints parity ratios for each group and stores them in
``artefacts/group_metrics.csv``.

The ``advanced_demo.ipynb`` notebook walks through these steps and shows the
additional plots.
