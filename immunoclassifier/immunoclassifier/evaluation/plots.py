#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Visualization — confusion matrices, UMAP predictions, benchmark charts.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    normalize: bool = True,
    figsize: tuple = (12, 10),
    cmap: str = "Blues",
    title: str = "Immune Cell Type Classification",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix for immune cell type classification.

    Parameters
    ----------
    y_true
        Ground truth labels
    y_pred
        Predicted labels
    labels
        Ordered list of class labels
    normalize
        If True, normalize by row (true label)
    figsize
        Figure size
    cmap
        Colormap
    title
        Plot title
    save_path
        If provided, save figure to this path

    Returns
    -------
    matplotlib Figure
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Cell Type", fontsize=12)
    ax.set_ylabel("True Cell Type", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_umap_predictions(
    adata,
    pred_key: str = "predicted_cell_type",
    true_key: str | None = "cell_type",
    figsize: tuple = (16, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot UMAP colored by predicted vs true cell types.

    Parameters
    ----------
    adata
        AnnData with UMAP coordinates and predictions in obs
    pred_key
        Column in obs with predicted labels
    true_key
        Column in obs with true labels (if available)
    figsize
        Figure size
    save_path
        If provided, save figure

    Returns
    -------
    matplotlib Figure
    """
    import scanpy as sc

    n_panels = 2 if true_key and true_key in adata.obs.columns else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    if n_panels == 1:
        axes = [axes]

    sc.pl.umap(adata, color=pred_key, ax=axes[0], show=False, title="Predicted")

    if n_panels == 2:
        sc.pl.umap(adata, color=true_key, ax=axes[1], show=False, title="Ground Truth")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved UMAP plot to {save_path}")

    return fig


def plot_benchmark_comparison(
    results: dict[str, dict[str, float]],
    metrics: list[str] = None,
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot benchmark comparison across methods.

    Parameters
    ----------
    results
        Dictionary mapping method name to metrics dict
    metrics
        Which metrics to plot (default: accuracy, balanced_accuracy, macro_f1)
    figsize
        Figure size
    save_path
        If provided, save figure

    Returns
    -------
    matplotlib Figure
    """
    if metrics is None:
        metrics = ["accuracy", "balanced_accuracy", "macro_f1"]

    # Build comparison dataframe
    rows = []
    for method, method_results in results.items():
        for metric in metrics:
            if metric in method_results:
                rows.append({
                    "Method": method,
                    "Metric": metric,
                    "Score": method_results[metric],
                })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x="Method", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Immune Cell Classification Benchmark", fontsize=14)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved benchmark plot to {save_path}")

    return fig

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
