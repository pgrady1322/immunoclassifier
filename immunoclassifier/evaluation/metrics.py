#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Classification metrics, per-class analysis, and rare cell evaluation.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    y_true
        Ground truth cell type labels
    y_pred
        Predicted cell type labels
    labels
        Ordered list of class labels

    Returns
    -------
    Dictionary with overall and per-class metrics
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "n_cells": len(y_true),
        "n_classes_true": len(np.unique(y_true)),
        "n_classes_pred": len(np.unique(y_pred)),
    }

    # Per-class metrics
    results["per_class"] = per_class_metrics(y_true, y_pred, labels)

    # Classification report as string
    results["report"] = classification_report(y_true, y_pred, zero_division=0)

    logger.info(
        f"Accuracy: {results['accuracy']:.4f}, "
        f"Balanced Acc: {results['balanced_accuracy']:.4f}, "
        f"Macro F1: {results['macro_f1']:.4f}"
    )

    return results


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-class precision, recall, F1, and support.

    Returns
    -------
    DataFrame with one row per cell type
    """
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    rows = []
    for label in labels:
        mask_true = y_true == label
        mask_pred = y_pred == label

        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        support = int(np.sum(mask_true))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "cell_type": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    df = pd.DataFrame(rows).set_index("cell_type")
    return df


def rare_cell_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: int = 100,
) -> dict[str, Any]:
    """
    Analyze classification performance on rare cell types.

    Rare cell types (< threshold cells) are common in immune data
    (e.g., pDCs, CD56bright NK cells) and are often misclassified.

    Parameters
    ----------
    y_true
        Ground truth labels
    y_pred
        Predicted labels
    threshold
        Cell count below which a type is considered "rare"

    Returns
    -------
    Dictionary with rare vs abundant cell type metrics
    """
    unique, counts = np.unique(y_true, return_counts=True)
    count_map = dict(zip(unique, counts, strict=False))

    rare_types = [ct for ct, n in count_map.items() if n < threshold]
    abundant_types = [ct for ct, n in count_map.items() if n >= threshold]

    # Rare cell metrics
    rare_mask = np.isin(y_true, rare_types)
    if rare_mask.sum() > 0:
        rare_acc = accuracy_score(y_true[rare_mask], y_pred[rare_mask])
        rare_f1 = f1_score(y_true[rare_mask], y_pred[rare_mask], average="macro", zero_division=0)
    else:
        rare_acc = rare_f1 = 0.0

    # Abundant cell metrics
    abund_mask = np.isin(y_true, abundant_types)
    if abund_mask.sum() > 0:
        abund_acc = accuracy_score(y_true[abund_mask], y_pred[abund_mask])
        abund_f1 = f1_score(y_true[abund_mask], y_pred[abund_mask], average="macro", zero_division=0)
    else:
        abund_acc = abund_f1 = 0.0

    results = {
        "rare_types": rare_types,
        "rare_n_types": len(rare_types),
        "rare_n_cells": int(rare_mask.sum()),
        "rare_accuracy": rare_acc,
        "rare_macro_f1": rare_f1,
        "abundant_types": abundant_types,
        "abundant_n_types": len(abundant_types),
        "abundant_n_cells": int(abund_mask.sum()),
        "abundant_accuracy": abund_acc,
        "abundant_macro_f1": abund_f1,
        "threshold": threshold,
    }

    logger.info(
        f"Rare cells (<{threshold}): {len(rare_types)} types, "
        f"acc={rare_acc:.4f}, F1={rare_f1:.4f} | "
        f"Abundant: {len(abundant_types)} types, "
        f"acc={abund_acc:.4f}, F1={abund_f1:.4f}"
    )

    return results

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
