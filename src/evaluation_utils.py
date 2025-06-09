from __future__ import annotations

from pathlib import Path
from typing import Callable

from .fairness import youden_threshold, four_fifths_ratio

__all__ = ["plot_or_load", "youden_thr", "four_fifths"]


def plot_or_load(plot_fn: Callable[[Path], None], path: str | Path) -> Path:
    """Run ``plot_fn`` to create ``path`` if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        plot_fn(p)
    return p


def youden_thr(estimator, X, y) -> float:
    """Return threshold maximising TPR minus FPR."""
    return youden_threshold(estimator, X, y)


def four_fifths(estimator, X, y, group_col: str, thr: float) -> float:
    """Return four-fifths ratio for ``group_col`` at ``thr``."""
    return four_fifths_ratio(estimator, X, y, group_col, thr)
