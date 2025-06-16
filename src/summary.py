from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import dataprep

__all__ = ["dataset_summary", "main"]


def dataset_summary(df: pd.DataFrame, target: str = "Loan_Status") -> str:
    """Return a short dataset overview string."""
    rows, cols = df.shape
    parts = [f"Rows: {rows}", f"Columns: {cols}"]
    if target in df.columns:
        counts = df[target].value_counts()
        total = counts.sum()
        stats = []
        for cls, cnt in counts.items():
            pct = cnt / total * 100
            stats.append(f"{cls}: {cnt} ({pct:.1f}%)")
        parts.append("Class balance: " + ", ".join(stats))
    return "\n".join(parts)


def main(args: list[str] | None = None) -> None:
    """CLI entry point printing dataset statistics."""
    parser = argparse.ArgumentParser(description="Print dataset statistics")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=dataprep.CSV_PATH,
        help="CSV dataset path",
    )
    parser.add_argument(
        "--target",
        default="Loan_Status",
        help="target column name",
    )
    ns = parser.parse_args(args)
    df = dataprep.clean(dataprep.load_raw(ns.data_path))
    print(dataset_summary(df, ns.target))


if __name__ == "__main__":
    main()
