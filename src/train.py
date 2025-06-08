from __future__ import annotations

import argparse
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from .models import logreg, cart


def main(args: list[str] | None = None) -> None:
    """CLI entry point training the logistic and tree models."""
    parser = argparse.ArgumentParser(description="Train model pipelines")
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        choices=["logreg", "cart"],
        help="models to train; defaults to both",
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
    ns = parser.parse_args(args)
    models = ns.model or ["logreg", "cart"]

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
        logreg.main(ns.data_path, sampler)
    if "cart" in models:
        cart.main(ns.data_path, sampler)


if __name__ == "__main__":
    main()
