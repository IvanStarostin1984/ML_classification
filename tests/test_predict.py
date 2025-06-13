from __future__ import annotations

import os
import sys
import subprocess
import sysconfig
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import joblib


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=20,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return df, pd.Series(y)


def test_cli_predict(tmp_path) -> None:
    df, y = _toy_data()
    model = LogisticRegression(max_iter=1000).fit(df, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )

    env = os.environ.copy()
    scripts_dir = Path(sysconfig.get_path("scripts"))
    env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")

    res = subprocess.run(
        [
            "mlcls-predict",
            "--model-path",
            str(model_path),
            "--data",
            str(data_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Predictions written" in res.stdout

    out = tmp_path / "predictions.csv"
    assert out.exists()
    preds = pd.read_csv(out)
    assert len(preds) == len(df)
