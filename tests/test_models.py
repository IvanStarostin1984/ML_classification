from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.logreg import train_from_df as train_logreg
from src.models.cart import train_from_df as train_cart
from src.models import logreg, cart
from src.features import FeatureEngineer
from src import dataprep


def _toy_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "income_annum": rng.normal(200_000, 50_000, n),
        "loan_amount": rng.normal(100_000, 20_000, n),
        "loan_term": rng.integers(6, 24, n),
        "cibil_score": rng.integers(600, 750, n),
        "education": rng.choice(["Graduate", "Not Graduate"], n),
        "self_employed": rng.choice(["Yes", "No"], n),
        "residential_assets_value": rng.uniform(50_000, 150_000, n),
        "commercial_assets_value": rng.uniform(0, 100_000, n),
        "luxury_assets_value": rng.uniform(0, 50_000, n),
        "bank_asset_value": rng.uniform(0, 50_000, n),
        "gender": rng.choice(["M", "F"], n),
        "married": rng.choice(["Yes", "No"], n),
        "property_area": rng.choice(["Urban", "Rural", "Semiurban"], n),
        "no_of_dependents": rng.integers(0, 4, n),
        "target": rng.integers(0, 2, n),
    })
    return df


def test_train_logreg() -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    auc = train_logreg(df, "target")
    assert 0 <= auc <= 1


def test_train_cart() -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    auc = train_cart(df, "target")
    assert 0 <= auc <= 1


def test_load_data_path(tmp_path) -> None:
    df = _toy_df().rename(columns={"target": "Loan_Status"})
    csv = tmp_path / "loan.csv"
    df.to_csv(csv, index=False)
    df_lr = logreg.load_data(csv)
    df_cart = cart.load_data(csv)
    assert "loan_status" in df_lr.columns
    assert "loan_status" in df_cart.columns
