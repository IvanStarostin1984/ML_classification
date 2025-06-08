"""ML classification utilities."""

from .features import FeatureEngineer
from .diagnostics import chi_square_tests, correlation_heatmap
from .preprocessing import build_preprocessor, safe_transform
from .selection import calculate_vif, tree_feature_selector
from .evaluate import evaluate_models
from .fairness import four_fifths_ratio, youden_threshold
from .calibration import calibrate_model

__all__ = [
    "FeatureEngineer",
    "chi_square_tests",
    "correlation_heatmap",
    "build_preprocessor",
    "safe_transform",
    "calculate_vif",
    "tree_feature_selector",
    "evaluate_models",
    "four_fifths_ratio",
    "youden_threshold",
    "calibrate_model",
]
