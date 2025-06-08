import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from src import dataprep  # noqa: E402


def test_load_raw(tmp_path):
  csv = tmp_path / 'file.csv'
  csv.write_text('A,B\n1,2\n')
  df = dataprep.load_raw(csv)
  assert df.shape == (1, 2)


def test_clean():
  df = pd.DataFrame({'A': [1, 1, None], 'Loan_Status': ['Y', 'N', 'Y']})
  cleaned = dataprep.clean(df)
  assert cleaned.shape == (2, 2)
  assert set(cleaned['Loan_Status']) <= {0, 1}
