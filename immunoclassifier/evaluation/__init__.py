"""Evaluation metrics and visualization for immune cell classification."""

from immunoclassifier.evaluation.metrics import (
    evaluate_predictions,
    per_class_metrics,
    rare_cell_analysis,
)
from immunoclassifier.evaluation.plots import (
    plot_confusion_matrix,
    plot_umap_predictions,
    plot_benchmark_comparison,
)

__all__ = [
    "evaluate_predictions",
    "per_class_metrics",
    "rare_cell_analysis",
    "plot_confusion_matrix",
    "plot_umap_predictions",
    "plot_benchmark_comparison",
]
