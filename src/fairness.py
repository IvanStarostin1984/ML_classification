from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def youden_threshold(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Return threshold maximising the Youden J statistic."""
    fpr, tpr, thr = roc_curve(y, estimator.predict_proba(X)[:, 1])
    return thr[np.argmax(tpr - fpr)]


def four_fifths_ratio(
    estimator, X: pd.DataFrame, y: pd.Series, group_col: str, thr: float
) -> float:
    """Return 4/5ths rule ratio of TPR across ``group_col``."""
    groups = X[group_col].astype(str)
    yhat = estimator.predict_proba(X)[:, 1] >= thr
    tprs = []
    for g in groups.unique():
        mask = groups == g
        positives = y[mask] == 1
        if positives.sum():
            tprs.append((yhat[mask] & positives).sum() / positives.sum())
    return min(tprs) / max(tprs) if len(tprs) > 1 else 1.0


def equal_opportunity_ratio(
    estimator, X: pd.DataFrame, y: pd.Series, group_col: str, thr: float
) -> float:
    """Return ratio of true positive rates across ``group_col``."""
    return four_fifths_ratio(estimator, X, y, group_col, thr)
