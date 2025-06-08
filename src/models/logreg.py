from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ..dataprep import clean
from ..features import FeatureEngineer
from ..preprocessing import build_preprocessor

from ..split import stratified_split

DATA_PATH = Path("data/raw/loan_approval_dataset.csv")
TARGET = "loan_status"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return cleaned and engineered DataFrame loaded from ``path``."""
    df = pd.read_csv(path)
    df = clean(df)
    return FeatureEngineer().transform(df)


def build_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    """Create preprocessing and logistic regression pipeline."""
    preproc = build_preprocessor(num_cols, cat_cols)
    model = LogisticRegression(max_iter=1000)
    return Pipeline([("prep", preproc), ("model", model)])


def train_from_df(
    df: pd.DataFrame,
    target: str = TARGET,
    artefact_path: Path | None = None,
) -> float:
    """Train model on ``df`` and return validation ROC-AUC."""
    train_df, val_df, _ = stratified_split(df, target)
    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_val = val_df.drop(columns=[target])
    y_val = val_df[target]
    cat_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(x_train, y_train)
    pred = pipe.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    if artefact_path:
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, artefact_path)
    return auc


def main() -> None:
    df = load_data()
    auc = train_from_df(df, artefact_path=Path("artefacts/logreg.joblib"))
    print(f"Validation ROC-AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
