from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.base import SamplerMixin
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from ..dataprep import clean
from ..features import FeatureEngineer
from ..preprocessing import build_preprocessor
from ..split import stratified_split

DATA_PATH = Path("data/raw/loan_approval_dataset.csv")
TARGET = "loan_status"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Return cleaned and engineered DataFrame loaded from ``path``."""
    df = pd.read_csv(Path(path))
    df = clean(df)
    return FeatureEngineer().transform(df)


def build_pipeline(
    cat_cols: list[str], num_cols: list[str], sampler: SamplerMixin | None = None
) -> Pipeline:
    """Create preprocessing and decision-tree pipeline."""
    preproc = build_preprocessor(num_cols, cat_cols)
    model = DecisionTreeClassifier(random_state=42)
    steps = [("prep", preproc)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


def train_from_df(
    df: pd.DataFrame,
    target: str = TARGET,
    artefact_path: Path | None = None,
    sampler: SamplerMixin | None = None,
) -> float:
    """Train model on ``df`` and return validation ROC-AUC."""
    train_df, val_df, _ = stratified_split(df, target)
    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_val = val_df.drop(columns=[target])
    y_val = val_df[target]
    cat_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    pipe = build_pipeline(cat_cols, num_cols, sampler)
    pipe.fit(x_train, y_train)
    pred = pipe.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    if artefact_path:
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, artefact_path)
    return auc


def grid_train_from_df(
    df: pd.DataFrame,
    target: str = TARGET,

    artefact_path: Path | None = None,
    sampler: SamplerMixin | None = None,
) -> float:
    """Train decision tree with grid search and return validation ROC-AUC."""
    train_df, val_df, _ = stratified_split(df, target)
    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_val = val_df.drop(columns=[target])
    y_val = val_df[target]
    cat_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]
    pipe = build_pipeline(cat_cols, num_cols, sampler)
    gs = GridSearchCV(
        pipe,
        {
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
        },
        cv=3,
        scoring="roc_auc",
    )
    gs.fit(x_train, y_train)
    pred = gs.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    if artefact_path:
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, artefact_path)
    return auc

    sampler: SamplerMixin | None = None,
) -> GridSearchCV:
    """Return fitted GridSearchCV on ``df`` using repeated CV."""
    x = df.drop(columns=[target])
    y = df[target]
    cat_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x.columns if c not in cat_cols]
    pipe = build_pipeline(cat_cols, num_cols, sampler)
    grid = {
        "model__max_depth": [None, 8, 15],
        "model__min_samples_leaf": [1, 5],
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(x, y)
    return gs



def main(
    data_path: str | Path = DATA_PATH, sampler: SamplerMixin | None = None
) -> None:
    df = load_data(data_path)
    auc = train_from_df(
        df, artefact_path=Path("artefacts/cart.joblib"), sampler=sampler
    )
    print(f"Validation ROC-AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
