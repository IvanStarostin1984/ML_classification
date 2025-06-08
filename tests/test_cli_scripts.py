from __future__ import annotations

import os
import sys
import subprocess
import sysconfig
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def _toy_df(n: int = 30) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=n,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    return pd.DataFrame(
        {
            "loan_amount": abs(x[:, 0]) * 100 + 100,
            "loan_term": (abs(x[:, 1]) * 10 + 10).astype(int),
            "cibil_score": abs(x[:, 2]) * 100 + 500,
            "Loan_Status": pd.Series(y).map({1: "Y", 0: "N"}),
            "education": ["Graduate"] * n,
            "self_employed": ["No"] * n,
            "group": [0] * (n // 2) + [1] * (n - n // 2),
        }
    )


def test_console_scripts(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )

    df = _toy_df()
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    df.to_csv(data_dir / "loan_approval_dataset.csv", index=False)

    env = os.environ.copy()
    scripts_dir = Path(sysconfig.get_path("scripts"))
    env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")

    eval_res = subprocess.run(
        ["mlcls-eval", "--group-col", "group"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "roc_auc" in eval_res.stdout

    train_res = subprocess.run(
        ["mlcls-train", "--model", "logreg"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Validation ROC-AUC" in train_res.stdout
