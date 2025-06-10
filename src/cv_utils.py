from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

__all__ = ["build_outer_iter", "nested_cv"]


def build_outer_iter(
    y: pd.Series,
    *,
    n_splits: int = 3,
    n_repeats: int = 2,
    bootstrap_iters: int = 100,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return outer CV iterator or bootstrap splits when minority <10."""
    if y.value_counts().min() >= 10:
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        dummy = np.zeros(len(y))
        return list(rskf.split(dummy, y))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    idx_pos = idx[y == 1]
    idx_neg = idx[y == 0]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(bootstrap_iters):
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        train = np.concatenate([boot_pos, boot_neg])
        oob = np.setdiff1d(idx, np.unique(train))
        splits.append((train, oob if oob.size else idx))
    return splits


def _build_pipeline(model, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([("prep", pre), ("model", model)])


def nested_cv(
    df: pd.DataFrame,
    target: str,
    model,
    grid: dict,
    scorers: dict,
    *,
    seed: int = 0,
    n_splits: int = 3,
    n_repeats: int = 2,
    bootstrap_iters: int = 100,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Run nested cross-validation with bootstrap fallback."""
    X = df.drop(columns=[target])
    y = df[target]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    pipe = _build_pipeline(model, cat_cols, num_cols)
    inner = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    outer = build_outer_iter(
        y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )
    gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", refit="roc_auc")
    res = cross_validate(gs, X, y, cv=outer, scoring=scorers, return_estimator=True)
    return res, X, y
