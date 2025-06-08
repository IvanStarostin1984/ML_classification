import pandas as pd
import pytest
import warnings
from src.selection import calculate_vif, tree_feature_selector


def test_vif_returns_series():
  df = pd.DataFrame({
    'a': [1.0, 2.0, 3.0],
    'b': [1.0, 2.0, 3.0],
    'c': [1.0, 1.0, 1.0],
  })
  res = calculate_vif(df, ['a', 'b'])
  assert res.index.tolist() == ['a', 'b']


def test_vif_no_warning():
  df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [2.0, 4.0, 6.0]})
  with warnings.catch_warnings(record=True) as w:
    calculate_vif(df, ['a', 'b'])
  assert not w


def test_tree_selector():
  X = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [4, 3, 2, 1]})
  y = pd.Series([0, 1, 0, 1])
  top = tree_feature_selector(X, y, n_estimators=10, top=1)
  assert len(top) == 1
