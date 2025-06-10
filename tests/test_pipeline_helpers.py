from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.pipeline_helpers import lr_steps, tree_steps, run_gs


def test_lr_and_tree_steps() -> None:
    prep = StandardScaler()
    assert lr_steps(prep, "passthrough") == [
        ("prep", prep),
        ("sampler", "passthrough"),
        ("model", None),
    ]
    assert tree_steps(prep, "sampler") == [
        ("prep", prep),
        ("sampler", "sampler"),
        ("model", None),
    ]


def test_run_gs_returns_cv() -> None:
    X, y = make_classification(
        n_samples=20, n_features=4, n_informative=3, n_redundant=0, random_state=0
    )
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    y_ser = pd.Series(y)
    steps = lr_steps(StandardScaler(), "passthrough")
    grid = {"model__C": [0.1, 1]}
    gs = run_gs(X_df, y_ser, steps, LogisticRegression(max_iter=1000), grid)
    assert hasattr(gs, "best_estimator_")
