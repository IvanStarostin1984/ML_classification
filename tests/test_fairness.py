import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.fairness import youden_threshold, four_fifths_ratio, equal_opportunity_ratio


def test_youden_threshold_range():
    X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 1, 0, 0]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegression().fit(X, y)
    thr = youden_threshold(model, X, y)
    assert 0.0 <= thr <= 1.0


def test_four_fifths_ratio_range():
    X = pd.DataFrame(
        {
            "feat": [0, 1, 0, 1, 0, 1, 0, 1],
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    y = pd.Series([1, 1, 0, 0, 1, 0, 1, 0])
    model = LogisticRegression().fit(X, y)
    thr = youden_threshold(model, X, y)
    ratio = four_fifths_ratio(model, X, y, "group", thr)
    assert 0.0 <= ratio <= 1.0


def test_equal_opportunity_ratio_alias():
    X = pd.DataFrame(
        {
            "feat": [0, 1, 0, 1, 0, 1, 0, 1],
            "group": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    y = pd.Series([1, 1, 0, 0, 1, 0, 1, 0])
    model = LogisticRegression().fit(X, y)
    thr = youden_threshold(model, X, y)
    ratio1 = four_fifths_ratio(model, X, y, "group", thr)
    ratio2 = equal_opportunity_ratio(model, X, y, "group", thr)
    assert ratio1 == ratio2
