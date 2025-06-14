Command-line usage
==================

Examples below assume the project is installed in editable mode::

   pip install -e .

Train models and store artefacts under ``artefacts/``::

   mlcls-train --model logreg
   mlcls-train -g  # grid search

Evaluate metrics and write ``artefacts/summary_metrics.csv``::

   mlcls-eval --group-col gender

Generate predictions and save them to ``predictions.csv`` (change
``--out`` to override)::

   mlcls-predict --model-path artefacts/logreg.joblib --data data/new.csv

The commands create the output paths in the current working directory.

Collect tables and figures for reporting::

   mlcls-report

The command gathers recent metrics and plots under ``report_artifacts/``. This
folder can be zipped and shared as a summary of the run.
