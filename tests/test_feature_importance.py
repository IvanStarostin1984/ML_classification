from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.logreg import train_from_df as train_logreg
from src.models.cart import train_from_df as train_cart
from src.features import FeatureEngineer
from src import dataprep
from src.feature_importance import logreg_coefficients, tree_feature_importances


def _toy_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
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
        }
    )
    return df


def test_logreg_coeff_csv(tmp_path) -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    model_fp = tmp_path / "lr.joblib"
    train_logreg(df, "target", artefact_path=model_fp)
    csv = tmp_path / "coef.csv"
    logreg_coefficients(model_fp, csv)
    assert csv.exists()


def test_logreg_shap_png(tmp_path) -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    model_fp = tmp_path / "lr.joblib"
    train_logreg(df, "target", artefact_path=model_fp)
    png = tmp_path / "shap.png"
    logreg_coefficients(model_fp, tmp_path / "coef.csv", shap_png_path=png, X=df)
    assert png.exists()


def test_cart_importance_csv(tmp_path) -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    model_fp = tmp_path / "cart.joblib"
    train_cart(df, "target", artefact_path=model_fp)
    csv = tmp_path / "imp.csv"
    tree_feature_importances(model_fp, csv)
    assert csv.exists()


def test_cart_shap_png(tmp_path) -> None:
    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    model_fp = tmp_path / "cart.joblib"
    train_cart(df, "target", artefact_path=model_fp)
    png = tmp_path / "cart_shap.png"
    tree_feature_importances(model_fp, tmp_path / "imp.csv", shap_png_path=png, X=df)
    assert png.exists()
