"""ColumnTransformer helpers."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import shapiro
import random

__all__ = [
    "build_preprocessor",
    "safe_transform",
    "_scaled_matrix",
    "_check_mu_sigma",
    "validate_prep",
]


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
    """Transform X_new dropping unseen columns."""
    if not isinstance(X_new, pd.DataFrame):
        raise TypeError("safe_transform expects a pandas DataFrame.")
    common = X_new.columns.intersection(preprocessor.feature_names_in_)
    extras = set(X_new.columns) - set(common)
    if extras:
        warnings.warn(f"dropped unseen columns at predict-time: {sorted(extras)}")
    return preprocessor.transform(X_new[common])


ROBUST_IQR_TOL = 0.02
SHAPIRO_MAX_ROWS = 5_000


def _scaled_matrix(
    prep: ColumnTransformer, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return transformed matrix and feature names."""
    Xs = safe_transform(prep, X)
    try:
        names = prep.get_feature_names_out()
    except AttributeError:
        names = np.arange(Xs.shape[1]).astype(str)
    return Xs, np.asarray(names)


def _check_mu_sigma(
    mat: np.ndarray, idx: np.ndarray, tol_mu: float = 1e-3, tol_sd: float = 1e-2
) -> bool:
    """Return True if columns mean ≈0 and sd ≈1."""
    mu = mat[:, idx].mean(0)
    sd = mat[:, idx].std(0, ddof=0)
    return (np.abs(mu) < tol_mu).all() and (np.abs(sd - 1) < tol_sd).all()


def validate_prep(
    prep: ColumnTransformer, X: pd.DataFrame, name: str, check_scale: bool = True
) -> None:
    """Raise if ``prep`` produces NaNs or deviates from unit scale."""
    Xs, names = _scaled_matrix(prep, X)
    if np.isnan(Xs).any() or np.isinf(Xs).any():
        raise ValueError(f"{name}: NaN/Inf produced by scaling.")

    if check_scale:
        prefix = np.array([n.split("__", 1)[0] if "__" in n else "" for n in names])
        idx_std = np.where(prefix == "std")[0]
        if idx_std.size and not _check_mu_sigma(Xs, idx_std):
            raise ValueError(f"{name}: std cols not μ≈0, σ≈1.")

        idx_rob = np.where(prefix == "rob")[0]
        if idx_rob.size:
            med = np.median(Xs[:, idx_rob], axis=0)
            iqr = np.percentile(Xs[:, idx_rob], 75, axis=0) - np.percentile(
                Xs[:, idx_rob], 25, axis=0
            )
            if (np.abs(med) > 1e-3).any() or (np.abs(iqr - 1) > ROBUST_IQR_TOL).any():
                raise ValueError(
                    f"{name}: robust cols not median≈0, IQR≈1±{ROBUST_IQR_TOL}."
                )

        idx_pow = np.where(prefix == "pow")[0]
        if idx_pow.size and not _check_mu_sigma(Xs, idx_pow):
            raise ValueError(f"{name}: power cols not μ≈0, σ≈1.")
        if idx_pow.size:
            pvals = []
            for i in idx_pow:
                col = Xs[:, i]
                if col.size > SHAPIRO_MAX_ROWS:
                    col = col[random.sample(range(col.size), SHAPIRO_MAX_ROWS)]
                pvals.append(shapiro(col)[1] if col.size >= 3 else 1.0)
            print(f"   PowerT median Shapiro-p = {np.median(pvals):.3f}")

    print(f"✅ {name} validated – shape {Xs.shape}")
