from __future__ import annotations

from typing import Sequence

from imblearn.base import SamplerMixin
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


__all__ = ["lr_steps", "tree_steps", "run_gs"]


def lr_steps(preprocessor, sampler: SamplerMixin | str) -> list[tuple[str, object]]:
    """Return pipeline steps for logistic regression."""
    return [("prep", preprocessor), ("sampler", sampler), ("model", None)]


def tree_steps(preprocessor, sampler: SamplerMixin | str) -> list[tuple[str, object]]:
    """Return pipeline steps for decision tree."""
    return [("prep", preprocessor), ("sampler", sampler), ("model", None)]


def run_gs(
    X,
    y,
    steps: Sequence[tuple[str, object]],
    estimator: BaseEstimator,
    grid: dict,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> GridSearchCV:
    """Fit ``GridSearchCV`` on ``X, y`` using ``steps`` and ``grid``."""
    pipe = Pipeline(steps)
    pipe.set_params(model=estimator)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    return gs
