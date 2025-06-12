import random
import numpy as np
import pandas as pd
from src.utils import (
    set_seeds,
    is_binary_numeric,
    zeros_like,
    dedup_pairs,
)


def test_set_seeds_reproducible() -> None:
    set_seeds(1)
    val1 = random.random()
    arr1 = np.random.rand(3)
    set_seeds(1)
    val2 = random.random()
    arr2 = np.random.rand(3)
    assert val1 == val2
    assert np.array_equal(arr1, arr2)


def test_is_binary_numeric() -> None:
    assert is_binary_numeric(pd.Series([0, 1, 1, 0]))
    assert is_binary_numeric(pd.Series([0.0, 1.0, np.nan]))
    assert not is_binary_numeric(pd.Series([0, 1, 2]))
    assert not is_binary_numeric(pd.Series(list("ab")))


def test_zeros_like() -> None:
    index = pd.Index(["a", "b", "c"])
    series = zeros_like(index)
    assert list(series.index) == list(index)
    assert series.tolist() == [0, 0, 0]


def test_dedup_pairs() -> None:
    merged = dedup_pairs([(1, 2)], [(2, 1), (3, 4)])
    assert merged == [(1, 2), (3, 4)]
