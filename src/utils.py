from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["set_seeds", "is_binary_numeric", "zeros_like", "dedup_pairs"]


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy and ``PYTHONHASHSEED``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def is_binary_numeric(series: pd.Series) -> bool:
    """Return ``True`` if numeric ``series`` contains only ``0``/``1`` values."""
    if series.dtype.kind not in "if":
        return False
    return set(series.dropna().unique()) <= {0, 1}


def zeros_like(index: pd.Index) -> pd.Series:
    """Return a zero-filled ``Series`` aligned with ``index``."""
    return pd.Series(0, index=index)


def dedup_pairs(old: Sequence[tuple], new: Sequence[tuple]) -> list[tuple]:
    """Merge ``old`` and ``new`` lists of 2-tuples, dropping duplicates."""

    seen: set[tuple] = set()
    merged: list[tuple] = []
    for pair in [*old, *new]:
        key = tuple(sorted(pair))
        if key not in seen:
            seen.add(key)
            merged.append(pair)
    return merged
