import hashlib
import numpy as np
from sklearn.model_selection import KFold

from src.manifest import sha256, shasum, save_folds


def test_checksum_helpers(tmp_path) -> None:
    fp = tmp_path / "f.txt"
    fp.write_text("hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256(fp) == expected
    assert shasum(fp) == expected[:12]


def test_save_folds(tmp_path) -> None:
    X = np.zeros((6, 1))
    y = np.array([0, 1, 0, 1, 0, 1])
    cv = KFold(n_splits=3)
    path = save_folds("foo", cv, X, y, out_dir=tmp_path)
    data = np.load(path)
    assert set(data.files) == {"fold_0", "fold_1", "fold_2"}
    assert path.exists()
