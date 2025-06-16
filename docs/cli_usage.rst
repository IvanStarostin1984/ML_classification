Command-line usage
==================

Examples below assume the project is installed in editable mode::

   pip install -e .

Train models and store artefacts under ``artefacts/``::

   mlcls-train --model logreg
   mlcls-train --model random_forest
   mlcls-train --model random_forest -g  # grid search
   mlcls-train --model gboost
   mlcls-train --model gboost -g  # grid search
   mlcls-train --model svm
   mlcls-train --model svm -g  # grid search

Evaluate metrics and write ``artefacts/summary_metrics.csv``::

   mlcls-eval --group-col gender

Generate predictions and save them to ``predictions.csv`` (change
``--out`` to override)::

   mlcls-predict --model-path artefacts/logreg.joblib --data data/new.csv

The commands create the output paths in the current working directory.

Collect tables and figures for reporting::

   mlcls-report

Create a checksum manifest::

   mlcls-manifest artefacts/*.csv

Show dataset statistics::

   mlcls-summary --data-path data/raw/loan_approval_dataset.csv

The command gathers recent metrics and plots under ``report_artifacts/``. This
folder can be zipped and shared as a summary of the run.

Local testing
-------------

Install the requirements before running the tests::

    pip install -r requirements.txt
    # or: conda env create -f environment.yml
