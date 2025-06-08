"""Fairness auditing helpers."""

from __future__ import annotations

import pandas as pd
from typing import Sequence

__all__ = ["four_fifths_ratio"]


def four_fifths_ratio(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    group: Sequence,
) -> float:
    """Return four-fifths rule TPR ratio for ``group``.

    Parameters
    ----------
    y_true:
        Binary ground truth labels.
    y_pred:
        Binary predictions.
    group:
        Protected attribute values used for grouping.
    """
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).reset_index(drop=True)
    group_s = pd.Series(group).astype(str).reset_index(drop=True)
    tprs: list[float] = []
    for g in group_s.unique():
        mask = group_s == g
        pos = (y_true_s[mask] == 1)
        n_pos = int(pos.sum())
        if n_pos == 0:
            tprs.append(1.0)
            continue
        tp = int((pos & (y_pred_s[mask] == 1)).sum())
        tprs.append(tp / n_pos)
    if len(tprs) < 2:
        return 1.0
    best = max(tprs)
    return min(tprs) / best if best else 1.0
