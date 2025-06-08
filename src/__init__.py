"""ML classification utilities."""

from .features import FeatureEngineer
from .diagnostics import (
    chi_square_tests,
    correlation_heatmap,
    roc_pr_boxplots,
    fairness_bar,
)
from .preprocessing import build_preprocessor, safe_transform
from .selection import calculate_vif, tree_feature_selector
from .evaluate import evaluate_models
from .fairness import four_fifths_ratio, youden_threshold
from .manifest import write_manifest

__all__ = [
    "FeatureEngineer",
    "chi_square_tests",
    "correlation_heatmap",
    "roc_pr_boxplots",
    "fairness_bar",
    "build_preprocessor",
    "safe_transform",
    "calculate_vif",
    "tree_feature_selector",
    "evaluate_models",
    "four_fifths_ratio",
    "youden_threshold",
    "write_manifest",
]
