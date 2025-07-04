from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import dataprep
from .cv_utils import nested_cv
from .fairness import (
    equal_opportunity_ratio,
    equalized_odds_diff,
    four_fifths_ratio,
    youden_threshold,
)

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
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    group_col: str | None,
    threshold: float | None,
) -> float:
    if group_col and group_col in X.columns:
        thr = threshold if threshold is not None else youden_threshold(estimator, X, y)
        return four_fifths_ratio(estimator, X, y, group_col, thr)
    return 1.0


def _equal_opp(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    group_col: str | None,
    threshold: float | None,
) -> float:
    if group_col and group_col in X.columns:
        thr = threshold if threshold is not None else youden_threshold(estimator, X, y)
        return equal_opportunity_ratio(estimator, X, y, group_col, thr)
    return 1.0


def _eq_odds_diff(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    group_col: str | None,
    threshold: float | None,
) -> float:
    if group_col and group_col in X.columns:
        thr = threshold if threshold is not None else youden_threshold(estimator, X, y)
        return equalized_odds_diff(estimator, X, y, group_col, thr)
    return 0.0


def evaluate_models(
    df: pd.DataFrame,
    target: str = "Loan_Status",
    group_col: str | None = None,
    csv_path: Path = Path("artefacts/summary_metrics.csv"),
    threshold: float | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Return nested-CV metrics and write ``csv_path``.

    ``threshold`` sets the probability cutoff used for group metrics. When it
    is ``None`` the Youden J statistic is used instead. ``models`` selects which
    pipelines to run.
    """
    df = dataprep.clean(df)

    all_models: dict[str, tuple[object, dict]] = {
        "logreg": (
            LogisticRegression(max_iter=1000, solver="liblinear"),
            {"model__C": [0.3, 1, 3], "model__penalty": ["l1", "l2"]},
        ),
        "cart": (
            DecisionTreeClassifier(random_state=42),
            {"model__max_depth": [None, 8, 15], "model__min_samples_leaf": [1, 5]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [50, 100], "model__max_depth": [None, 10]},
        ),
        "gboost": (
            GradientBoostingClassifier(random_state=42),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
        ),
        "svm": (
            SVC(probability=True),
            {"model__kernel": ["linear", "rbf"], "model__C": [0.1, 1.0]},
        ),
    }

    selected = models or list(all_models.keys())
    rows = []
    for name in selected:
        model, grid = all_models[name]
        res, X, y = nested_cv(
            df,
            target,
            model,
            grid,
            SCORERS,
            seed=0,
            n_splits=3,
            n_repeats=2,
            bootstrap_iters=100,
        )
        rows.append(
            {
                "model": name,
                "roc_auc": res["test_roc_auc"].mean(),
                "pr_auc": res["test_pr_auc"].mean(),
                "f1": res["test_f1"].mean(),
                "recall": res["test_recall"].mean(),
                "specificity": res["test_specificity"].mean(),
                "bal_acc": res["test_bal_acc"].mean(),
                "fairness": _four_fifths(
                    res["estimator"][0], X, y, group_col, threshold
                ),
                "equal_opp": _equal_opp(
                    res["estimator"][0], X, y, group_col, threshold
                ),
                "eq_odds": _eq_odds_diff(
                    res["estimator"][0], X, y, group_col, threshold
                ),
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
    parser.add_argument(
        "--threshold",
        type=float,
        help="probability cutoff for fairness metrics",
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logreg", "cart", "random_forest", "gboost", "svm"],
        default=None,
        help="models to evaluate (default: all)",
    )
    ns = parser.parse_args(args)

    from . import dataprep

    df = dataprep.clean(dataprep.load_raw())
    metrics = evaluate_models(
        df, group_col=ns.group_col, threshold=ns.threshold, models=ns.models
    )
    print(metrics)


if __name__ == "__main__":
    main()
