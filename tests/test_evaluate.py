from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification

from src.evaluate import evaluate_models


def _toy_df() -> pd.DataFrame:
    x, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["target"] = y
    df["group"] = [0] * 25 + [1] * 25
    return df


def test_evaluate_models(tmp_path) -> None:
    df = _toy_df()
    csv_path = tmp_path / "summary_metrics.csv"
    res = evaluate_models(df, target="target", group_col="group", csv_path=csv_path)
    assert isinstance(res, pd.DataFrame)
    assert csv_path.exists()
