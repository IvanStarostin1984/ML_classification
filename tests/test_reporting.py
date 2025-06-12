from __future__ import annotations

import pandas as pd

from src.reporting import flatten_cv, flatten_metrics
from src.report_helpers import conf_matrix_summary, group_metrics


def test_flatten_cv(tmp_path) -> None:
    idx = ["DT", "LR"]
    cols = pd.MultiIndex.from_tuples(
        [
            ("roc_auc", "mean"),
            ("roc_auc", "std"),
            ("pr_auc", "mean"),
        ]
    )
    df = pd.DataFrame([[0.8, 0.05, 0.6], [0.75, 0.1, 0.55]], index=idx, columns=cols)
    csv = tmp_path / "cv.csv"
    df.to_csv(csv)

    flat = flatten_cv(csv)
    assert list(flat.columns) == ["roc_auc_mean", "roc_auc_std", "pr_auc_mean"]
    assert flat.loc["DT", "roc_auc_mean"] == 0.8
    assert flat.loc["LR", "pr_auc_mean"] == 0.55


def test_flatten_metrics() -> None:
    md = {
        "roc_auc": [0.1, 0.2],
        "acc": 0.93456,
        "detail": {"tp": 10.0},
        "bootstrap_iters": 3,
    }
    flat = flatten_metrics(md)
    assert flat["roc_auc_low"] == 0.1
    assert flat["roc_auc_high"] == 0.2
    assert flat["acc"] == 0.9346
    assert flat["detail_tp"] == 10
    assert flat["bootstrap_iters"] == 3


def test_conf_matrix_summary() -> None:
    s = conf_matrix_summary([1, 0, 1, 0], [1, 0, 1, 1])
    assert s.startswith("Recall=1.000")
    assert "BalAcc=0.750" in s
    assert s.endswith("F1=0.800")


def test_group_metrics() -> None:
    df = pd.DataFrame({"loan_status": [1, 0, 1, 0], "sex": ["A", "A", "B", "B"]})
    prob = pd.Series([0.9, 0.3, 0.2, 0.8])
    res = group_metrics(df, "sex", prob, 0.5)
    a = res[res["group"] == "A"].iloc[0]
    b = res[res["group"] == "B"].iloc[0]
    assert (a["n_pos"], a["n_neg"], a["TPR"]) == (1, 1, 1.0)
    assert (b["n_pos"], b["n_neg"], b["TPR"]) == (1, 1, 0.0)
