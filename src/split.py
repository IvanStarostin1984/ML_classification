from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ["stratified_split", "random_split", "time_split"]


def stratified_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation and test splits stratified by ``target``."""
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target],
        random_state=random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[target],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return random train/test split (optionally stratified)."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify] if stratify else None,
    )
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


def time_split(
    df: pd.DataFrame,
    date_col: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return chronological train/test split by ``date_col``."""
    if date_col not in df.columns:
        raise ValueError(f"{date_col} not in DataFrame")
    if df[date_col].isna().any():
        raise ValueError(f"{date_col} contains NaNs")
    df_sorted = df.sort_values(date_col, kind="mergesort")
    if not df_sorted[date_col].is_monotonic_increasing:
        raise ValueError(f"{date_col} is not strictly increasing")
    cut = int(round((1 - test_size) * len(df_sorted)))
    return (
        df_sorted.iloc[:cut].reset_index(drop=True),
        df_sorted.iloc[cut:].reset_index(drop=True),
    )
