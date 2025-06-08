from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.metrics import make_scorer, recall_score
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from .fairness import youden_threshold, four_fifths_ratio

SPECIFICITY = make_scorer(recall_score, pos_label=0)
SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "specificity": SPECIFICITY,
    "bal_acc": "balanced_accuracy",
}
INNER_CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)
OUTER_CV = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=1)


def _bootstrap_splits(y: pd.Series, n_iter: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    idx_pos = idx[y == 1]
    idx_neg = idx[y == 0]
    splits = []
    for _ in range(n_iter):
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        train = np.concatenate([boot_pos, boot_neg])
        oob = np.setdiff1d(idx, np.unique(train))
        splits.append((train, oob if oob.size else idx))
    return splits


def _build_pipeline(model, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([("prep", pre), ("model", model)])


def _run_nested(
    df: pd.DataFrame, target: str, model, grid: dict
) -> tuple[dict, pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    pipe = _build_pipeline(model, cat_cols, num_cols)
    gs = GridSearchCV(pipe, grid, cv=INNER_CV, scoring="roc_auc", refit="roc_auc")
    outer = OUTER_CV if y.value_counts().min() >= 10 else _bootstrap_splits(y)
    res = cross_validate(gs, X, y, cv=outer, scoring=SCORERS, return_estimator=True)
    return res, X, y


def _fairness(estimator, X: pd.DataFrame, y: pd.Series, group_col: str | None) -> float:
    if group_col and group_col in X.columns:
        thr = youden_threshold(estimator, X, y)
        return four_fifths_ratio(estimator, X, y, group_col, thr)
    return 1.0


def evaluate_models(
    df: pd.DataFrame,
    target: str = "Loan_Status",
    group_col: str | None = None,
    csv_path: Path = Path("artefacts/summary_metrics.csv"),
) -> pd.DataFrame:
    """Return nested-CV metrics for both models and write ``csv_path``."""
    lr_res, X, y = _run_nested(
        df,
        target,
        LogisticRegression(max_iter=1000, solver="liblinear"),
        {"model__C": [0.3, 1, 3], "model__penalty": ["l1", "l2"]},
    )
    dt_res, _, _ = _run_nested(
        df,
        target,
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [None, 8, 15], "model__min_samples_leaf": [1, 5]},
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
                "fairness": _fairness(res["estimator"][0], X, y, group_col),
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
