from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.shap_utils import compute_shap_values, plot_shap_summary


def test_compute_shap_values_shape() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"])
    y = rng.integers(0, 2, size=8)
    model = LogisticRegression().fit(X, y)
    shap_df = compute_shap_values(model, X)
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == X.shape


def test_plot_shap_summary(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(8, 3)), columns=["a", "b", "c"])
    y = rng.integers(0, 2, size=8)
    model = LogisticRegression().fit(X, y)
    png = tmp_path / "summary.png"
    plot_shap_summary(model, X, png)
    assert png.exists()
