from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)


__all__ = ["eval_metrics", "eval_at", "show_metrics", "folds_df"]


def eval_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], y_pred: Iterable[int], suffix: str
) -> dict:
    """Return performance metrics with ``suffix`` appended to keys."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        f"ROC_AUC{suffix}": roc_auc_score(y_true, y_prob),
        f"PR_AUC{suffix}": average_precision_score(y_true, y_prob),
        f"Brier{suffix}": brier_score_loss(y_true, y_prob),
        f"F1{suffix}": f1_score(y_true, y_pred),
        f"Recall{suffix}": recall_score(y_true, y_pred),
        f"Specificity{suffix}": tn / (tn + fp),
        f"BalAcc{suffix}": balanced_accuracy_score(y_true, y_pred),
    }


def eval_at(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict:
    """Compute metrics at ``threshold``."""
    y_pred = (pd.Series(y_prob) >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
    }


def show_metrics(
    label: str, y_true: Iterable[int], y_prob: Iterable[float], y_pred: Iterable[int]
) -> None:
    """Print test-set metrics in a single formatted line."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    print(
        f"{label:5s} ROC={roc_auc_score(y_true, y_prob):.3f}  "
        f"PR={average_precision_score(y_true, y_prob):.3f}  "
        f"F1={f1_score(y_true, y_pred):.3f}  "
        f"Rec={recall_score(y_true, y_pred):.3f}  "
        f"Spec={spec:.3f}  "
        f"BalAcc={balanced_accuracy_score(y_true, y_pred):.3f}"
    )


def folds_df(res: dict, model: str) -> pd.DataFrame:
    """Return tidy DataFrame of cross-validation folds."""
    return pd.DataFrame(
        {
            "roc_auc": res["test_roc_auc"],
            "pr_auc": res["test_pr_auc"],
            "brier": res["test_brier"],
        }
    ).assign(model=model)
