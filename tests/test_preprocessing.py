import pandas as pd
import pytest
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.preprocessing import (
    build_preprocessor,
    safe_transform,
    _scaled_matrix,
    _check_mu_sigma,
    validate_prep,
)


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


def test_scaled_matrix_names_and_shape():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pre = build_preprocessor(["num"], ["cat"])
    pre.fit(df, [0, 1, 0])
    mat, names = _scaled_matrix(pre, df)
    assert mat.shape[0] == 3
    assert len(names) == mat.shape[1]


def test_check_mu_sigma_true_false():
    mat = np.column_stack(
        [
            np.array([-1.0, 1.0, -1.0, 1.0]),
            np.linspace(0, 1, 4),
        ]
    )
    idx = np.array([0])
    assert _check_mu_sigma(mat, idx)
    assert not _check_mu_sigma(mat, np.array([1]))


def test_validate_prep_success_and_failure():
    df = pd.DataFrame({"num": [0.0, 1.0, 2.0], "cat": ["a", "b", "a"]})
    pre = ColumnTransformer(
        [
            ("std", Pipeline([("sc", StandardScaler())]), ["num"]),
            (
                "cat",
                Pipeline([("enc", OneHotEncoder(handle_unknown="ignore"))]),
                ["cat"],
            ),
        ]
    )
    pre.fit(df, [0, 1, 0])
    validate_prep(pre, df, "ok")

    df_bad = pd.DataFrame({"num": [1.0, 1.0, 1.0]})
    pre_bad = ColumnTransformer(
        [("std", Pipeline([("sc", StandardScaler())]), ["num"])]
    )
    pre_bad.fit(df_bad, [0, 0, 0])
    with pytest.raises(ValueError):
        validate_prep(pre_bad, df_bad, "fail")
