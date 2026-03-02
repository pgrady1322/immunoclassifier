#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Evaluation subpackage — metrics and plotting.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from immunoclassifier.evaluation.metrics import (
    evaluate_predictions,
    per_class_metrics,
    rare_cell_analysis,
)
from immunoclassifier.evaluation.plots import (
    plot_benchmark_comparison,
    plot_confusion_matrix,
    plot_umap_predictions,
)

__all__ = [
    "evaluate_predictions",
    "per_class_metrics",
    "rare_cell_analysis",
    "plot_confusion_matrix",
    "plot_umap_predictions",
    "plot_benchmark_comparison",
]

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
