import pandas as pd
import pytest
from src import dataprep


def test_load_raw(tmp_path):
    csv = tmp_path / "file.csv"
    csv.write_text("A,B\n1,2\n")
    df = dataprep.load_raw(csv)
    assert df.shape == (1, 2)


def test_load_raw_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataprep.load_raw(tmp_path / "missing.csv")


def test_clean():
    df = pd.DataFrame({"A": [1, 1, None], "Loan_Status": ["Y", "N", "Y"]})
    cleaned = dataprep.clean(df)
    assert cleaned.shape == (2, 2)
    assert set(cleaned["Loan_Status"]) <= {0, 1}
