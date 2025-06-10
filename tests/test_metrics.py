import pandas as pd
from src.metrics import eval_metrics, eval_at, show_metrics, folds_df


def test_eval_metrics() -> None:
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.3, 0.2, 0.8]
    y_pred = [int(p >= 0.5) for p in y_prob]
    res = eval_metrics(y_true, y_prob, y_pred, "@050")
    assert res["ROC_AUC@050"] == 0.5
    assert res["PR_AUC@050"] == 0.75
    assert round(res["Brier@050"], 3) == 0.345
    assert res["Specificity@050"] == 0.5


def test_eval_at_and_show(capsys) -> None:
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.3, 0.2, 0.8]
    metrics = eval_at(y_true, y_prob, 0.5)
    assert metrics["F1"] == 0.5
    y_pred = [int(p >= 0.5) for p in y_prob]
    show_metrics("LR", y_true, y_prob, y_pred)
    out = capsys.readouterr().out
    assert "ROC=0.500" in out
    assert "BalAcc=0.500" in out


def test_folds_df_shape() -> None:
    res = {
        "test_roc_auc": [0.7, 0.8],
        "test_pr_auc": [0.6, 0.7],
        "test_brier": [0.2, 0.3],
    }
    df = folds_df(res, "LR")
    assert list(df.columns) == ["roc_auc", "pr_auc", "brier", "model"]
    assert len(df) == 2
    assert set(df["model"]) == {"LR"}
