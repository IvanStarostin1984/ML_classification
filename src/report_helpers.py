from __future__ import annotations

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
import pandas as pd

__all__ = ["conf_matrix_summary", "group_metrics"]


def conf_matrix_summary(y_true, y_pred) -> str:
    """Return recall, balanced accuracy and F1 summary string."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return (
        f"Recall={tp/(tp+fn):.3f} "
        f"BalAcc={balanced_accuracy_score(y_true, y_pred):.3f} "
        f"F1={f1_score(y_true, y_pred):.3f}"
    )


def group_metrics(
    df: pd.DataFrame, col: str, prob: pd.Series, thr: float
) -> pd.DataFrame:
    """Return TPR per group at threshold ``thr``."""
    groups = df[col].astype(str).fillna("NA")
    rows = []
    for g in groups.unique():
        mask = groups == g
        yhat = prob[mask] >= thr
        tn, fp, fn, tp = confusion_matrix(df["loan_status"][mask], yhat).ravel()
        tpr = tp / (tp + fn) if tp + fn else 0
        rows.append((g, int(tp + fn), int(tn + fp), tpr))
    return pd.DataFrame(rows, columns=["group", "n_pos", "n_neg", "TPR"])
