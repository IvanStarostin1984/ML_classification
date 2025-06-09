from __future__ import annotations

import pandas as pd

from src.reporting import flatten_cv, flatten_metrics


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
