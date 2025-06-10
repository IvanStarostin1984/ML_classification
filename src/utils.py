from __future__ import annotations

import os
import random
import numpy as np

__all__ = ["set_seeds"]


def set_seeds(seed: int = 42) -> None:
    """Seed Python, NumPy and ``PYTHONHASHSEED``."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
