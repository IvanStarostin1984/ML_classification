"""ColumnTransformer helpers."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = ["build_preprocessor", "safe_transform"]


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Return basic preprocessing ColumnTransformer."""
    num_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def safe_transform(preprocessor: ColumnTransformer, X_new: pd.DataFrame) -> np.ndarray:
    """Transform ``X_new`` with safety checks."""
    if not isinstance(X_new, pd.DataFrame):
        raise TypeError("safe_transform expects a pandas DataFrame.")
    expected = set(preprocessor.feature_names_in_)
    present = set(X_new.columns)
    missing = expected - present
    if missing:
        raise ValueError(f"missing columns for transform: {sorted(missing)}")
    extras = present - expected
    if extras:
        warnings.warn(f"dropped unseen columns at predict-time: {sorted(extras)}")
    return preprocessor.transform(X_new[list(expected)])
