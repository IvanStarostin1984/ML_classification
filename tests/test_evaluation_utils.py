from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation_utils import plot_or_load, youden_thr, four_fifths
from src.fairness import youden_threshold, four_fifths_ratio


def _dummy_plot(path):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(path)
    plt.close()


def test_plot_or_load(tmp_path):
    fp = tmp_path / "plot.png"
    calls = []

    def _plot(p):
        calls.append(1)
        _dummy_plot(p)

    out = plot_or_load(_plot, fp)
    assert out == fp
    assert fp.exists()
    assert len(calls) == 1

    out = plot_or_load(_plot, fp)
    assert out == fp
    assert len(calls) == 1


def test_alias_helpers():
    X = pd.DataFrame({"a": [0, 1, 0, 1], "g": [0, 0, 0, 0]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    thr = youden_thr(model, X, y)
    assert thr == youden_threshold(model, X, y)
    ratio = four_fifths(model, X, y, "g", thr)
    assert ratio == four_fifths_ratio(model, X, y, "g", thr)
