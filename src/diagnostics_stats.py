"""Statistical diagnostics for contingency tables."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, MonteCarloMethod

__all__ = [
    "_need_exact",
    "_cramers_v",
    "_cochran_armitage",
    "_safe_chi2",
    "_fmt_p",
    "_annotate",
]

# number of Monte Carlo samples for chi-square fallback
MC_N = 5_000


def _need_exact(exp: np.ndarray) -> bool:
    """Return True if expected frequencies warrant Monte Carlo simulation."""
    flat = exp.ravel()
    return (flat < 1).any() or (flat < 5).sum() / flat.size > 0.20


def _cramers_v(chi2: float, tbl: np.ndarray) -> float:
    """Return Cramér's V effect size for a contingency table."""
    r, c = tbl.shape
    return math.sqrt(max(chi2, 0) / (tbl.sum() * (min(r, c) - 1)))


def _cochran_armitage(ct: pd.DataFrame) -> Tuple[float, float]:
    """Return Cochran–Armitage trend test Z and p-value for a 2×k table."""
    if ct.shape[0] != 2:
        return math.nan, math.nan
    scores = np.arange(1, ct.shape[1] + 1, dtype=float)
    n1j = ct.iloc[1].to_numpy(float)
    n_j = ct.sum(axis=0).to_numpy(float)
    n1, n = n1j.sum(), n_j.sum()
    if n == 0:
        return math.nan, math.nan
    t = float(np.dot(scores, n1j))
    mean_t = n1 * float(np.dot(scores, n_j)) / n
    var_t = (
        n1
        * (n - n1)
        * float(np.dot(n_j, (scores - float(np.dot(scores, n_j)) / n) ** 2))
        / (n * (n - 1))
    )
    if var_t == 0:
        return math.nan, math.nan
    z = (t - mean_t) / math.sqrt(var_t)
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def _safe_chi2(
    ct: pd.DataFrame,
    need_mc: bool,
    rng: np.random.Generator,
) -> Tuple[float, float, int, str]:
    """Return chi-square test with Monte Carlo or 0.5 adjustment fallback."""

    try:
        if need_mc:
            if "method" in chi2_contingency.__code__.co_varnames:
                res = chi2_contingency(
                    ct,
                    correction=False,
                    method=MonteCarloMethod(n_resamples=MC_N),
                )
                return (
                    res.statistic,
                    res.pvalue,
                    res.dof,
                    f"chi2 (MC {MC_N:,})",
                )
            else:
                n = int(ct.to_numpy().sum())
                expected = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / n
                chi2 = ((ct - expected) ** 2 / expected).to_numpy().sum()
                probs = (expected / n).ravel()
                sims = rng.multinomial(n, probs, size=MC_N).reshape(MC_N, *ct.shape)
                sim_chi2 = ((sims - expected) ** 2 / expected).sum(axis=(1, 2))
                p = (sim_chi2 >= chi2).mean()
                dof = (ct.shape[0] - 1) * (ct.shape[1] - 1)
                return chi2, float(p), dof, f"chi2 (MC {MC_N:,})"
        chi2, p, dof, _ = chi2_contingency(ct, correction=False, lambda_=None)
        return chi2, p, dof, "chi2"
    except ValueError:
        adj = ct.to_numpy(float) + 0.5
        chi2, p, dof, _ = chi2_contingency(adj, correction=False)
        return chi2, p, dof, "chi2 (+0.5 adj)"


def _fmt_p(p: float, thr: float = 1e-6) -> str:
    """Format a p-value string, showing ``<thr`` when below ``thr``."""
    if pd.isna(p):
        return "NA"
    return f"<{thr:.1e}" if p < thr else f"{p:.4f}"


def _annotate(ax) -> None:
    """Annotate bar plot with integer heights if supported."""
    if not hasattr(ax, "bar_label"):
        return
    for cont in ax.containers:
        data = getattr(cont, "datavalues", [patch.get_height() for patch in cont])
        ax.bar_label(cont, labels=[int(v) for v in data], fontsize=7, padding=2)
