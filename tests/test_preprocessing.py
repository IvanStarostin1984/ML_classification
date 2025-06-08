from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
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
