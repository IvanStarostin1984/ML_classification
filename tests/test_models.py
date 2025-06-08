from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from src.models.logreg import train_from_df as train_logreg
from src.models.cart import train_from_df as train_cart


def _toy_df() -> pd.DataFrame:
    x, y = make_classification(n_samples=100, n_features=5, random_state=0)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["target"] = y
    return df


def test_train_logreg() -> None:
    df = _toy_df()
    auc = train_logreg(df, "target")
    assert 0 <= auc <= 1


def test_train_cart() -> None:
    df = _toy_df()
    auc = train_cart(df, "target")
    assert 0 <= auc <= 1
