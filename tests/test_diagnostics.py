from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.diagnostics import chi_square_tests, correlation_heatmap


def test_chi_square_runs():
  df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F'],
    'approved': [1, 0, 1, 0],
  })
  res = chi_square_tests(df, 'approved')
  assert not res.empty


def test_heatmap_returns_ax():
  df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2, 3, 4],
  })
  ax = correlation_heatmap(df)
  assert hasattr(ax, 'figure')
