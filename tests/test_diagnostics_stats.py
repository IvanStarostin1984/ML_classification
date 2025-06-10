import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import chi2_contingency

from src.diagnostics_stats import (
    _need_exact,
    _cramers_v,
    _cochran_armitage,
    _safe_chi2,
    _fmt_p,
    _annotate,
)


def test_need_exact():
    ct = pd.DataFrame([[10, 10], [10, 10]])
    _, _, _, exp = chi2_contingency(ct, correction=False)
    assert not _need_exact(exp)

    ct_small = pd.DataFrame([[5, 1], [1, 5]])
    _, _, _, exp_small = chi2_contingency(ct_small, correction=False)
    assert _need_exact(exp_small)


def test_cramers_v():
    ct = pd.DataFrame([[10, 0], [0, 10]])
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    assert _cramers_v(chi2, ct.to_numpy()) == 1.0


def test_cochran_armitage():
    ct = pd.DataFrame([[10, 10, 10], [0, 10, 20]])
    z, p = _cochran_armitage(ct)
    assert z > 3
    assert p < 0.01


def test_safe_chi2_adjustment():
    ct = pd.DataFrame([[0, 0], [10, 10]])
    chi2, p, dof, note = _safe_chi2(ct, False, default_rng(0))
    assert "adj" in note
    assert dof == 1
    assert 0 <= p <= 1


def test_safe_chi2_montecarlo():
    ct = pd.DataFrame([[1, 2], [2, 3]])
    chi2, p, dof, note = _safe_chi2(ct, True, default_rng(1))
    assert "MC" in note
    assert 0 <= p <= 1


def test_fmt_p_and_annotate():
    assert _fmt_p(0.03) == "0.0300"
    assert _fmt_p(1e-7) == "<1.0e-06"
    assert _fmt_p(float("nan")) == "NA"

    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2])
    _annotate(ax)
    assert ax.texts
    plt.close(fig)
