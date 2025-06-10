from __future__ import annotations

import os
import random
import numpy as np
import pandas as pd

__all__ = ["set_seeds", "is_binary_numeric"]


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
