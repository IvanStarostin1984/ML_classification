from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .shap_utils import compute_shap_values, plot_shap_summary

__all__ = ["logreg_coefficients", "tree_feature_importances"]


def _feature_names(pipe, size: int) -> list[str]:
    prep = pipe.named_steps.get("prep")
    if hasattr(prep, "get_feature_names_out"):
        return list(prep.get_feature_names_out())
    return [f"f{i}" for i in range(size)]


def _bar_chart(names: list[str], values: np.ndarray, path: Path, xlabel: str) -> None:
    fig, ax = plt.subplots()
    ax.barh(names, values)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def logreg_coefficients(
    model_path: str | Path,
    csv_path: str | Path = Path("artefacts/logreg_coefficients.csv"),
    png_path: str | Path | None = None,
    shap_csv_path: str | Path | None = None,
    shap_png_path: str | Path | None = None,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save logistic-regression coefficients and odds ratios."""
    pipe = joblib.load(Path(model_path))
    coef = pipe.named_steps["model"].coef_.ravel()
    names = _feature_names(pipe, coef.size)
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef)})
    df = df.assign(abs_coef=lambda d: d.coef.abs()).sort_values(
        "abs_coef", ascending=False
    )
    csv = Path(csv_path)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="abs_coef").to_csv(csv, index=False)
    if shap_csv_path:
        if X is None:
            raise ValueError("X must be provided when shap_csv_path is set")
        prep = pipe.named_steps.get("prep")
        X_trans = prep.transform(X) if prep else X
        shap_df = compute_shap_values(
            pipe.named_steps["model"], X_trans, feature_names=names
        )
        shap_csv = Path(shap_csv_path)
        shap_csv.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_csv, index=False)
    if shap_png_path:
        if X is None:
            raise ValueError("X must be provided when shap_png_path is set")
        prep = pipe.named_steps.get("prep")
        X_trans = prep.transform(X) if prep else X
        plot_shap_summary(pipe.named_steps["model"], X_trans, shap_png_path)
    if png_path:
        _bar_chart(
            df.feature.tolist(), df.odds_ratio.to_numpy(), Path(png_path), "Odds ratio"
        )
    return df


def tree_feature_importances(
    model_path: str | Path,
    csv_path: str | Path = Path("artefacts/cart_importances.csv"),
    png_path: str | Path | None = None,
    shap_csv_path: str | Path | None = None,
    shap_png_path: str | Path | None = None,
    X: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Save decision-tree feature importances."""
    pipe = joblib.load(Path(model_path))
    imps = pipe.named_steps["model"].feature_importances_
    names = _feature_names(pipe, imps.size)
    df = pd.DataFrame({"feature": names, "importance": imps}).sort_values(
        "importance", ascending=False
    )
    csv = Path(csv_path)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    if shap_csv_path:
        if X is None:
            raise ValueError("X must be provided when shap_csv_path is set")
        prep = pipe.named_steps.get("prep")
        X_trans = prep.transform(X) if prep else X
        shap_df = compute_shap_values(
            pipe.named_steps["model"], X_trans, feature_names=names
        )
        shap_csv = Path(shap_csv_path)
        shap_csv.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(shap_csv, index=False)
    if shap_png_path:
        if X is None:
            raise ValueError("X must be provided when shap_png_path is set")
        prep = pipe.named_steps.get("prep")
        X_trans = prep.transform(X) if prep else X
        plot_shap_summary(pipe.named_steps["model"], X_trans, shap_png_path)
    if png_path:
        _bar_chart(
            df.feature.tolist(), df.importance.to_numpy(), Path(png_path), "Importance"
        )
    return df
