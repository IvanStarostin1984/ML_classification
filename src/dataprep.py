"""Utilities for loading and cleaning the Kaggle loan dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

CSV_PATH = Path('data/raw/loan_approval_dataset.csv')


def load_raw(path: str | Path = CSV_PATH) -> pd.DataFrame:
  """Return the raw dataset as a ``DataFrame``."""
  return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates/NA and normalise target column."""
    df = df.drop_duplicates().dropna().copy()
    df.columns = df.columns.str.strip()

    if "loan_status" in df.columns and "Loan_Status" not in df.columns:
        df = df.rename(columns={"loan_status": "Loan_Status"})

    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].astype(str).str.strip()
        mapping = {
            "Y": 1,
            "N": 0,
            "Approved": 1,
            "Rejected": 0,
        }
        df["Loan_Status"] = (
            df["Loan_Status"].map(mapping).fillna(df["Loan_Status"]).astype("Int64")
        )

    return df
