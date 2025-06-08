import pandas as pd
import pytest
from src.preprocessing import build_preprocessor, safe_transform


def test_build_and_transform():
  df = pd.DataFrame({
    'num': [1.0, 2.0],
    'cat': ['a', 'b'],
  })
  pre = build_preprocessor(['num'], ['cat'])
  pre.fit(df, [0, 1])
  X = safe_transform(pre, df)
  assert X.shape[0] == 2


def test_safe_transform_missing_column():
    df = pd.DataFrame({'num': [1.0], 'cat': ['a']})
    pre = build_preprocessor(['num'], ['cat'])
    pre.fit(df, [0])
    df_test = pd.DataFrame({'num': [2.0]})
    with pytest.raises(ValueError):
        safe_transform(pre, df_test)
