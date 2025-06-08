from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from .fairness import youden_threshold, four_fifths_ratio

SCORERS = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
INNER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
OUTER_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)


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
    res = cross_validate(gs, X, y, cv=OUTER_CV, scoring=SCORERS, return_estimator=True)
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
        df, target, LogisticRegression(max_iter=1000), {"model__C": [0.5, 1.5]}
    )
    dt_res, _, _ = _run_nested(
        df,
        target,
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [None, 3]},
    )
    rows = []
    for name, res in [("logreg", lr_res), ("cart", dt_res)]:
        rows.append(
            {
                "model": name,
                "roc_auc": res["test_roc_auc"].mean(),
                "pr_auc": res["test_pr_auc"].mean(),
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
