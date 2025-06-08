"""Utilities for loading and cleaning the Kaggle loan dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

CSV_PATH = Path('data/raw/loan_approval_dataset.csv')


def load_raw(path: str | Path = CSV_PATH) -> pd.DataFrame:
    """Return the raw dataset as a ``DataFrame``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"No data loaded from {p}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
  """Basic cleaning: drop duplicates/NA and normalise target column."""
  df = df.drop_duplicates().dropna().copy()
  if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0}).astype('Int64')
  return df
