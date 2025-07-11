from __future__ import annotations

import argparse
from pathlib import Path

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from .models import cart, gradient_boosting, logreg, random_forest, svm


def main(args: list[str] | None = None) -> None:
    """CLI entry point training the logistic and tree models."""
    parser = argparse.ArgumentParser(description="Train model pipelines")
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        choices=["logreg", "cart", "random_forest", "gboost", "svm"],
        help="models to train; defaults to all",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=logreg.DATA_PATH,
        help="CSV dataset path",
    )
    parser.add_argument(
        "--sampler",
        choices=[
            "smote",
            "smotenc",
            "smoten",
            "randomover",
            "randomunder",
            "smotetomek",
        ],
        default=None,
        help="optional imbalance sampler",
    )
    parser.add_argument(
        "--grid-search",
        "-g",
        action="store_true",
        help="use grid search to tune hyperparameters",
    )
    ns = parser.parse_args(args)
    models = ns.model or ["logreg", "cart", "random_forest", "gboost", "svm"]

    sampler_map = {
        "smote": SMOTE,
        "smotenc": SMOTENC,
        "smoten": SMOTEN,
        "randomover": RandomOverSampler,
        "randomunder": RandomUnderSampler,
        "smotetomek": SMOTETomek,
    }
    sampler = sampler_map[ns.sampler]() if ns.sampler else None

    if "logreg" in models:
        if ns.grid_search:
            df = logreg.load_data(ns.data_path)
            auc = logreg.grid_train_from_df(
                df,
                artefact_path=Path("artefacts/logreg.joblib"),
                sampler=sampler,
            )
            print(f"Validation ROC-AUC: {auc:.3f}")
        else:
            logreg.main(ns.data_path, sampler)
    if "cart" in models:
        if ns.grid_search:
            df = cart.load_data(ns.data_path)
            gs = cart.grid_train_from_df(df, sampler=sampler)
            print(f"Validation ROC-AUC: {gs.best_score_:.3f}")
        else:
            cart.main(ns.data_path, sampler)
    if "random_forest" in models:
        if ns.grid_search:
            df = random_forest.load_data(ns.data_path)
            gs = random_forest.grid_train_from_df(df, sampler=sampler)
            print(f"Validation ROC-AUC: {gs.best_score_:.3f}")
        else:
            random_forest.main(ns.data_path, sampler)
    if "gboost" in models:
        if ns.grid_search:
            df = gradient_boosting.load_data(ns.data_path)
            gs = gradient_boosting.grid_train_from_df(df, sampler=sampler)
            print(f"Validation ROC-AUC: {gs.best_score_:.3f}")
        else:
            gradient_boosting.main(ns.data_path, sampler)
    if "svm" in models:
        if ns.grid_search:
            df = svm.load_data(ns.data_path)
            gs = svm.grid_train_from_df(df, sampler=sampler)
            print(f"Validation ROC-AUC: {gs.best_score_:.3f}")
        else:
            svm.main(ns.data_path, sampler)


if __name__ == "__main__":
    main()
