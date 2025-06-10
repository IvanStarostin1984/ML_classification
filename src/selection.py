"""Feature selection helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from sklearn.ensemble import ExtraTreesClassifier

__all__ = ["calculate_vif", "vif_prune", "tree_feature_selector"]


def calculate_vif(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return variance inflation factors for numeric columns."""
    if df.shape[0] <= 3:
        return pd.Series([1.0] * len(cols), index=cols)
    arr = df[cols].to_numpy(float)
    if np.linalg.matrix_rank(arr) < arr.shape[1]:
        return pd.Series([float("inf")] * arr.shape[1], index=cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore"):
            vals = []
            for i in range(arr.shape[1]):
                try:
                    vals.append(vif(arr, i))
                except ValueError:
                    vals.append(float("inf"))
    return pd.Series(vals, index=cols)


def vif_prune(
    df: pd.DataFrame, cols: list[str], cap: float
) -> tuple[list[str], pd.Series]:
    """Return columns kept after iterative VIF pruning and their VIFs."""

    cols = list(cols)
    while True:
        if len(cols) <= 2:
            return cols, calculate_vif(df, cols)
        vifs = calculate_vif(df, cols)
        if vifs.max() <= cap:

        if len(cols) < 2:
            return cols, pd.Series([np.nan] * len(cols), index=cols)

        vifs = calculate_vif(df, cols)

        if vifs.max() <= cap or len(cols) <= 2:
            return cols, vifs
        cols.remove(vifs.replace(np.inf, 1e12).idxmax())


        if len(cols) == 2 and not np.isfinite(vifs).all():

            return cols, vifs

        if vifs.max() <= cap:
            return cols, vifs

        cols.remove(vifs.idxmax())



def tree_feature_selector(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    top: int = 10,
) -> list[str]:
    """Select important features using an ExtraTrees classifier."""
    clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=0)
    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns)
    return imp.nlargest(top).index.tolist()
