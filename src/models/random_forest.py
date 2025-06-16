from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.base import SamplerMixin
from imblearn.pipeline import Pipeline

from ..dataprep import clean, CSV_PATH
from ..features import FeatureEngineer
from ..preprocessing import build_preprocessor, validate_prep
from ..pipeline_helpers import tree_steps, run_gs
from ..split import stratified_split

DATA_PATH = CSV_PATH
TARGET = "loan_status"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Return cleaned and engineered DataFrame loaded from ``path``."""
    df = pd.read_csv(Path(path))
    df = clean(df)
    return FeatureEngineer().transform(df)


def build_pipeline(
    cat_cols: list[str], num_cols: list[str], sampler: SamplerMixin | None = None
) -> Pipeline:
    """Create preprocessing and random-forest pipeline."""
    preproc = build_preprocessor(num_cols, cat_cols)
    model = RandomForestClassifier(random_state=42)
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
    pipe.named_steps["prep"].fit(x_train, y_train)
    validate_prep(pipe.named_steps["prep"], x_train, "random_forest")
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
):
    """Return fitted GridSearchCV and optionally save best model."""
    x, y = df.drop(columns=[target]), df[target]
    cat_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in x.columns if c not in cat_cols]
    preproc = build_preprocessor(num_cols, cat_cols)
    preproc.fit(x, y)
    validate_prep(preproc, x, "random_forest")
    steps = tree_steps(preproc, sampler or "passthrough")
    grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 10],
        "model__min_samples_leaf": [1, 5],
        "model__min_samples_split": [2, 10],
        "model__class_weight": [None, "balanced"],
    }
    gs = run_gs(x, y, steps, RandomForestClassifier(random_state=42), grid)
    if artefact_path:
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(gs.best_estimator_, artefact_path)
    return gs


def main(
    data_path: str | Path = DATA_PATH, sampler: SamplerMixin | None = None
) -> None:
    df = load_data(data_path)
    auc = train_from_df(
        df, artefact_path=Path("artefacts/random_forest.joblib"), sampler=sampler
    )
    print(f"Validation ROC-AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
