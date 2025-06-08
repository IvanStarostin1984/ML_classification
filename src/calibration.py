from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import ClassifierMixin
import pandas as pd

from .models import logreg


def calibrate_model(
    estimator: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Return fitted calibration wrapper for ``estimator``."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be sigmoid or isotonic")
    cal = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    return cal.fit(X, y)


def _plot_curve(
    model: ClassifierMixin, X: pd.DataFrame, y: pd.Series, path: Path
) -> None:
    """Save calibration curve plot to ``path``."""
    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X)[:, 1], n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main(args: list[str] | None = None) -> None:
    """CLI entry calibrating saved models."""
    parser = argparse.ArgumentParser(description="Calibrate saved classifiers")
    parser.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ns = parser.parse_args(args)

    df = logreg.load_data()
    X = df.drop(columns=[logreg.TARGET])
    y = df[logreg.TARGET]

    artefacts = Path("artefacts")
    for name in ["logreg", "cart"]:
        model_path = artefacts / f"{name}.joblib"
        if not model_path.exists():
            continue
        est = joblib.load(model_path)
        cal = calibrate_model(est, X, y, ns.method)
        joblib.dump(cal, artefacts / f"{name}_calibrated.joblib")
        _plot_curve(cal, X, y, artefacts / f"{name}_calibration.png")


if __name__ == "__main__":
    main()
