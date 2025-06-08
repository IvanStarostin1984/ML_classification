"""Diagnostics for categorical and numeric features."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

__all__ = ["chi_square_tests", "correlation_heatmap"]


def chi_square_tests(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return chi-square p-values for categorical columns vs target."""
    records = []
    for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
        if col == target:
            continue
        ct = pd.crosstab(df[col], df[target])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(ct, correction=False)
        records.append({"feature": col, "chi2": chi2, "dof": dof, "p_value": p})
    return pd.DataFrame(records).sort_values("p_value")


def correlation_heatmap(df: pd.DataFrame, numeric: list[str] | None = None) -> plt.Axes:
    """Plot a Spearman correlation heatmap."""
    cols = numeric or df.select_dtypes("number").columns.tolist()
    corr = df[cols].corr("spearman").abs()
    ax = sns.heatmap(corr, cmap="viridis", square=True)
    ax.set_title("Absolute Spearman correlation")
    return ax
