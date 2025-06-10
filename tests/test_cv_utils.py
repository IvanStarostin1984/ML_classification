from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src import dataprep
from src.cv_utils import build_outer_iter, nested_cv
from src.evaluate import SCORERS


def _df(n_pos: int, n_neg: int) -> pd.DataFrame:
    total = n_pos + n_neg
    x, y = make_classification(
        n_samples=total,
        n_features=4,
        weights=[n_neg / total, n_pos / total],
        random_state=0,
    )
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["Loan_Status"] = pd.Series(y)
    return dataprep.clean(df)


def test_build_outer_iter_regular() -> None:
    df = _df(15, 15)
    splits = build_outer_iter(df["Loan_Status"], bootstrap_iters=5)
    assert len(splits) == 6
    tr, te = splits[0]
    assert len(tr) + len(te) == len(df)


def test_build_outer_iter_bootstrap() -> None:
    df = _df(18, 2)
    splits = build_outer_iter(df["Loan_Status"], bootstrap_iters=3)
    assert len(splits) == 3
    for tr, _ in splits:
        assert len(tr) == len(df)


def test_nested_cv_bootstrap_fallback() -> None:
    df = _df(18, 2)
    res, _, _ = nested_cv(
        df,
        "Loan_Status",
        LogisticRegression(max_iter=1000, solver="liblinear"),
        {"model__C": [1]},
        SCORERS,
        bootstrap_iters=2,
    )
    assert len(res["test_roc_auc"]) == 2
