from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.calibration import calibrate_model
from src import dataprep
from src.features import FeatureEngineer
from src.models import logreg


def test_calibrate_model_simple() -> None:
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    model = LogisticRegression().fit(X, y)
    cal = calibrate_model(model, X, y)
    assert hasattr(cal, "predict_proba")


def test_calibrate_model_isotonic_fitted() -> None:
  X, y = make_classification(n_samples=30, n_features=4, random_state=1)
  model = LogisticRegression().fit(X, y)
  cal = calibrate_model(model, X, y, method="isotonic")
  assert hasattr(cal, "calibrated_classifiers_")


def test_calibrate_model_invalid_method() -> None:
  X, y = make_classification(n_samples=20, n_features=4, random_state=2)
  model = LogisticRegression().fit(X, y)
  with pytest.raises(ValueError):
    calibrate_model(model, X, y, method="bad")


def _toy_df(n: int = 40) -> pd.DataFrame:
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
            "property_area": rng.choice(["Urban", "Rural"], n),
            "no_of_dependents": rng.integers(0, 4, n),
            "Loan_Status": pd.Series(rng.integers(0, 2, n)).map({1: "Y", 0: "N"}),
        }
    )
    return df


def test_cli_calibration(tmp_path) -> None:
    df = _toy_df()
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    df.to_csv(data_dir / "loan_approval_dataset.csv", index=False)

    df_clean = dataprep.clean(df)
    df_fe = FeatureEngineer().transform(df_clean)
    artefacts = tmp_path / "artefacts"
    artefacts.mkdir()
    logreg.train_from_df(
        df_fe, target="loan_status", artefact_path=artefacts / "logreg.joblib"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.calibration",
        ],
        cwd=tmp_path,
        check=True,
        env=env,
    )

    plot = artefacts / "logreg_calibration.png"
    assert plot.exists()
