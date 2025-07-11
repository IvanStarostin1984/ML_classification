import pandas as pd
import numpy as np
import warnings
from src.selection import calculate_vif, tree_feature_selector, vif_prune


def test_vif_returns_series():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],
            "c": [1.0, 1.0, 1.0],
        }
    )
    res = calculate_vif(df, ["a", "b"])
    assert res.index.tolist() == ["a", "b"]


def test_vif_no_warning():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    with warnings.catch_warnings(record=True) as w:
        calculate_vif(df, ["a", "b"])
    assert not w


def test_tree_selector():
    X = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1])
    top = tree_feature_selector(X, y, n_estimators=10, top=1)
    assert len(top) == 1


def test_vif_prune_drops_highest_vif():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],  # perfectly correlated with a
            "c": [1.0, 0.0, 1.0, 0.0],
        }
    )
    cols, vifs = vif_prune(df, ["a", "b", "c"], cap=5)
    assert set(cols) == {"c", "a"} or set(cols) == {"c", "b"}
    assert (vifs <= 5).all()


def test_vif_prune_no_drop_when_below_cap():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 2.2, 3.3]})
    cols, vifs = vif_prune(df, ["x", "y"], cap=100)
    assert cols == ["x", "y"]
    assert list(vifs.index) == ["x", "y"]


def test_vif_prune_single_column():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    cols, vifs = vif_prune(df, ["x"], cap=5)
    assert cols == ["x"]
    assert vifs.isna().all()


def test_vif_prune_two_infinite_vifs():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    cols, vifs = vif_prune(df, ["a", "b"], cap=1000)
    assert cols == ["a", "b"]
    assert vifs.index.tolist() == ["a", "b"]
    assert not vifs.replace([np.inf, -np.inf], np.nan).notna().all()
