from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, recall_score

from .cv_utils import nested_cv

from .fairness import youden_threshold, four_fifths_ratio, equal_opportunity_ratio

SPECIFICITY = make_scorer(recall_score, pos_label=0)
SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "specificity": SPECIFICITY,
    "bal_acc": "balanced_accuracy",
}


def _four_fifths(
    estimator, X: pd.DataFrame, y: pd.Series, group_col: str | None
) -> float:
    if group_col and group_col in X.columns:
        thr = youden_threshold(estimator, X, y)
        return four_fifths_ratio(estimator, X, y, group_col, thr)
    return 1.0


def _equal_opp(
    estimator, X: pd.DataFrame, y: pd.Series, group_col: str | None
) -> float:
    if group_col and group_col in X.columns:
        thr = youden_threshold(estimator, X, y)
        return equal_opportunity_ratio(estimator, X, y, group_col, thr)
    return 1.0


def evaluate_models(
    df: pd.DataFrame,
    target: str = "Loan_Status",
    group_col: str | None = None,
    csv_path: Path = Path("artefacts/summary_metrics.csv"),
) -> pd.DataFrame:
    """Return nested-CV metrics for both models and write ``csv_path``."""
    lr_res, X, y = nested_cv(
        df,
        target,
        LogisticRegression(max_iter=1000, solver="liblinear"),
        {"model__C": [0.3, 1, 3], "model__penalty": ["l1", "l2"]},
        SCORERS,
        seed=0,
        n_splits=3,
        n_repeats=2,
        bootstrap_iters=100,
    )
    dt_res, _, _ = nested_cv(
        df,
        target,
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [None, 8, 15], "model__min_samples_leaf": [1, 5]},
        SCORERS,
        seed=0,
        n_splits=3,
        n_repeats=2,
        bootstrap_iters=100,
    )
    rows = []
    for name, res in [("logreg", lr_res), ("cart", dt_res)]:
        rows.append(
            {
                "model": name,
                "roc_auc": res["test_roc_auc"].mean(),
                "pr_auc": res["test_pr_auc"].mean(),
                "f1": res["test_f1"].mean(),
                "recall": res["test_recall"].mean(),
                "specificity": res["test_specificity"].mean(),
                "bal_acc": res["test_bal_acc"].mean(),
                "fairness": _four_fifths(res["estimator"][0], X, y, group_col),
                "equal_opp": _equal_opp(res["estimator"][0], X, y, group_col),
            }
        )
    out = pd.DataFrame(rows).round(3)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return out


def main(args: list[str] | None = None) -> None:
    """CLI entry point evaluating both models on the cleaned dataset."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--group-col",
        help="optional fairness group column",
        default=None,
    )
    ns = parser.parse_args(args)

    from . import dataprep

    df = dataprep.clean(dataprep.load_raw())
    metrics = evaluate_models(df, group_col=ns.group_col)
    print(metrics)


if __name__ == "__main__":
    main()
