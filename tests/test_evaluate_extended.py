from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src import dataprep, evaluate
from src.cv_utils import nested_cv


def _df() -> pd.DataFrame:
    x, y = make_classification(n_samples=40, n_features=4, random_state=1)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["Loan_Status"] = pd.Series(y).map({1: "Y", 0: "N"})
    return dataprep.clean(df)


def test_extended_metrics_and_grid() -> None:
    df = _df()
    metrics = evaluate.evaluate_models(df)
    for col in ["f1", "recall", "specificity", "bal_acc"]:
        assert col in metrics.columns
    res, _, _ = nested_cv(
        df,
        "Loan_Status",
        LogisticRegression(max_iter=1000, solver="liblinear"),
        {"model__C": [0.3, 1, 3], "model__penalty": ["l1", "l2"]},
        evaluate.SCORERS,
    )
    assert len(res["estimator"][0].cv_results_["params"]) > 1
