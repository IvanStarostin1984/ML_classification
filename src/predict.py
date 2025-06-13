from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main(args: list[str] | None = None) -> None:
    """CLI entry point applying a trained model to new data."""
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="joblib model path",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="CSV of features for prediction",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("predictions.csv"),
        help="output CSV path",
    )
    ns = parser.parse_args(args)

    model = joblib.load(ns.model_path)
    df = pd.read_csv(ns.data)

    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(df)[:, 1]
    else:
        preds = model.predict(df)

    out_df = pd.DataFrame({"prediction": preds})
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ns.out, index=False)
    print(f"Predictions written to {ns.out}")


if __name__ == "__main__":
    main()
