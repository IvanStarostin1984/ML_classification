import random
import numpy as np
from src.utils import set_seeds


def test_set_seeds_reproducible() -> None:
    set_seeds(1)
    val1 = random.random()
    arr1 = np.random.rand(3)
    set_seeds(1)
    val2 = random.random()
    arr2 = np.random.rand(3)
    assert val1 == val2
    assert np.array_equal(arr1, arr2)
