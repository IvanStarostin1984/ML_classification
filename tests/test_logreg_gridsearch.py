from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.models import logreg
from src.models.logreg import grid_train_from_df
from src import dataprep
from src.features import FeatureEngineer


def _toy_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(2)
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


def test_grid_train_from_df_runs() -> None:
    grid = []
    for blk in logreg._LOGREG_PARAM_GRID_BASE:
        new_blk = blk.copy()
        new_blk["sampler"] = ["passthrough", "passthrough"]
        grid.append(new_blk)
    assert len(list(ParameterGrid(grid))) > 1

    df = _toy_df()
    df = dataprep.clean(df)
    df = FeatureEngineer().transform(df)
    auc = grid_train_from_df(df, "target")
    assert 0 <= auc <= 1
