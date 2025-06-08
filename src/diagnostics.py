"""Diagnostics for categorical and numeric features."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

__all__ = [
    "chi_square_tests",
    "correlation_heatmap",
    "roc_pr_boxplots",
    "fairness_bar",
]


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


def roc_pr_boxplots(folds: pd.DataFrame) -> plt.Axes:
    """Return boxplots for ROC-AUC and PR-AUC scores."""
    models = folds["model"].unique().tolist()
    roc = [folds.loc[folds["model"] == m, "roc_auc"] for m in models]
    pr = [folds.loc[folds["model"] == m, "pr_auc"] for m in models]
    pos1 = list(range(1, len(models) + 1))
    pos2 = list(range(len(models) + 2, len(models) * 2 + 2))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot(roc, positions=pos1, labels=models)
    ax.boxplot(pr, positions=pos2, labels=models)
    ticks = [sum(pos1) / len(pos1), sum(pos2) / len(pos2)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Cross-validated score")
    return ax


def fairness_bar(metrics: pd.DataFrame) -> plt.Axes:
    """Return bar chart for fairness ratios."""
    models = metrics["model"].unique().tolist()
    ratios = [metrics.loc[metrics["model"] == m, "fairness"].mean() for m in models]
    fig, ax = plt.subplots(figsize=(2.6, 3))
    ax.bar(models, ratios, color=sns.color_palette()[: len(models)])
    ax.axhline(0.8, color="r", ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("4/5ths TPR ratio")
    return ax
