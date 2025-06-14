from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

__all__ = ["compute_shap_values"]


def compute_shap_values(
    model: Any, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """Return a DataFrame of SHAP values for ``model`` on ``X``."""
    explainer = shap.Explainer(model, X)
    values = explainer(X).values
    if values.ndim == 3:
        values = values.sum(axis=1)
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            feature_names = [f"f{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, columns=feature_names)

