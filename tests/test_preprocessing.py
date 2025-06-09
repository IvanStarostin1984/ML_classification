import pandas as pd
import pytest
from src.preprocessing import build_preprocessor, safe_transform


def test_build_and_transform():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "cat": ["a", "b"],
        }
    )
    pre = build_preprocessor(["num"], ["cat"])
    pre.fit(df, [0, 1])
    X = safe_transform(pre, df)
    assert X.shape[0] == 2


def test_safe_transform_type_error():
    df = pd.DataFrame({"num": [1, 2]})
    pre = build_preprocessor(["num"], [])
    pre.fit(df, [0, 1])
    try:
        safe_transform(pre, [1, 2])
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


def test_safe_transform_extra_col_warning():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "cat": ["a", "b"],
        }
    )
    pre = build_preprocessor(["num"], ["cat"])
    pre.fit(df, [0, 1])
    df_extra = df.assign(extra=[3, 4])
    with pytest.warns(UserWarning):
        X = safe_transform(pre, df_extra)
    assert X.shape == (2, 3)
