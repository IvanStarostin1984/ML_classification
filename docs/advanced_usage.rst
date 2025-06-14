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
the output table.

Fairness checks
---------------

Use the evaluation command with ``--group-col`` to compute group metrics like
statistical parity::

   mlcls-eval --group-col gender

The ``advanced_demo.ipynb`` notebook walks through these steps and shows the
additional plots.
