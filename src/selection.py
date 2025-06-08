"""Feature selection helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)
from sklearn.ensemble import ExtraTreesClassifier

__all__ = ["calculate_vif", "tree_feature_selector"]


def calculate_vif(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return variance inflation factors for numeric columns."""
    arr = df[cols].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore"):
            vals = [vif(arr, i) for i in range(arr.shape[1])]
    return pd.Series(vals, index=cols)


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
