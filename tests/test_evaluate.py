from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def _toy_df() -> pd.DataFrame:
    x, y = make_classification(n_samples=50, n_features=4, random_state=0)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["Loan_Status"] = pd.Series(y).map({1: "Y", 0: "N"})
    df["group"] = [0] * 25 + [1] * 25
    return df


def test_cli_evaluate(tmp_path) -> None:
    df = _toy_df()
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    df.to_csv(data_dir / "loan_approval_dataset.csv", index=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--group-col", "group"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "roc_auc" in result.stdout
    summary = tmp_path / "artefacts" / "summary_metrics.csv"
    assert summary.exists()
