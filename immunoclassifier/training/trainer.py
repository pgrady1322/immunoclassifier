#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmunoClassifier v0.1.0

Model training, benchmarking, and cross-validation.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import anndata as ad
from sklearn.model_selection import StratifiedKFold

from immunoclassifier.models.base import BaseClassifier
from immunoclassifier.models import (
    LogisticClassifier,
    XGBoostClassifier,
    ScVIClassifier,
    GNNClassifier,
)
from immunoclassifier.evaluation.metrics import evaluate_predictions, rare_cell_analysis

logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    "logistic": LogisticClassifier,
    "xgboost": XGBoostClassifier,
    "scvi": ScVIClassifier,
    "gnn": GNNClassifier,
}


class Trainer:
    """
    Unified training and benchmarking orchestrator.

    Supports:
    - Training individual models
    - Cross-validation
    - Multi-model benchmarking
    - Cross-dataset evaluation
    """

    def __init__(
        self,
        output_dir: str = "results",
        label_key: str = "cell_type",
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_key = label_key
        self.random_state = random_state
        self.results = {}

    def train_model(
        self,
        model_name: str,
        adata: ad.AnnData,
        model_kwargs: Optional[Dict] = None,
        train_kwargs: Optional[Dict] = None,
    ) -> tuple:
        """
        Train a single model and return it with metrics.

        Parameters
        ----------
        model_name
            Key from MODEL_REGISTRY: 'logistic', 'xgboost', 'scvi', 'gnn'
        adata
            Preprocessed AnnData with cell type labels
        model_kwargs
            Arguments passed to model constructor
        train_kwargs
            Arguments passed to model.train()

        Returns
        -------
        Tuple of (trained model, training metrics dict)
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")

        model_kwargs = model_kwargs or {}
        train_kwargs = train_kwargs or {}

        logger.info(f"Training {model_name}...")
        start_time = time.time()

        model = MODEL_REGISTRY[model_name](**model_kwargs)
        metrics = model.train(adata, label_key=self.label_key, **train_kwargs)

        elapsed = time.time() - start_time
        metrics["training_time_seconds"] = elapsed

        logger.info(f"{model_name} trained in {elapsed:.1f}s")
        return model, metrics

    def benchmark(
        self,
        adata_train: ad.AnnData,
        adata_test: Optional[ad.AnnData] = None,
        models: Optional[List[str]] = None,
        model_configs: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple models on the same data.

        Parameters
        ----------
        adata_train
            Training data
        adata_test
            Held-out test data. If None, uses train/val split.
        models
            List of model names to benchmark. Default: all
        model_configs
            Per-model configuration overrides

        Returns
        -------
        Dictionary mapping model name to full evaluation results
        """
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        model_configs = model_configs or {}

        results = {}

        for model_name in models:
            try:
                config = model_configs.get(model_name, {})
                model, train_metrics = self.train_model(
                    model_name, adata_train, model_kwargs=config
                )

                # Evaluate on test set
                if adata_test is not None:
                    y_true = adata_test.obs[self.label_key].values
                    y_pred = model.predict(adata_test)
                    eval_results = evaluate_predictions(y_true, y_pred)
                    eval_results["training_metrics"] = train_metrics
                    eval_results["rare_cell_analysis"] = rare_cell_analysis(y_true, y_pred)
                else:
                    eval_results = {"training_metrics": train_metrics}

                results[model_name] = eval_results

                # Save model
                model_path = self.output_dir / "models" / model_name
                model_path.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path / "model"))

                logger.info(f"✓ {model_name} complete")

            except Exception as e:
                logger.error(f"✗ {model_name} failed: {e}")
                results[model_name] = {"error": str(e)}

        self.results = results
        return results

    def cross_validate(
        self,
        model_name: str,
        adata: ad.AnnData,
        n_folds: int = 5,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Stratified k-fold cross-validation.

        Parameters
        ----------
        model_name
            Model to cross-validate
        adata
            Full dataset
        n_folds
            Number of CV folds
        model_kwargs
            Model constructor arguments

        Returns
        -------
        Dictionary with per-fold and aggregated metrics
        """
        model_kwargs = model_kwargs or {}

        y = adata.obs[self.label_key].values
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            logger.info(f"Fold {fold + 1}/{n_folds}")

            adata_train = adata[train_idx].copy()
            adata_test = adata[test_idx].copy()

            model, _ = self.train_model(model_name, adata_train, model_kwargs=model_kwargs)

            y_true = adata_test.obs[self.label_key].values
            y_pred = model.predict(adata_test)
            fold_eval = evaluate_predictions(y_true, y_pred)
            fold_results.append(fold_eval)

        # Aggregate
        aggregate = {
            "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "std_accuracy": np.std([r["accuracy"] for r in fold_results]),
            "mean_balanced_accuracy": np.mean([r["balanced_accuracy"] for r in fold_results]),
            "std_balanced_accuracy": np.std([r["balanced_accuracy"] for r in fold_results]),
            "mean_macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
            "std_macro_f1": np.std([r["macro_f1"] for r in fold_results]),
            "n_folds": n_folds,
            "per_fold": fold_results,
        }

        logger.info(
            f"CV Results ({n_folds}-fold): "
            f"Acc={aggregate['mean_accuracy']:.4f}±{aggregate['std_accuracy']:.4f}, "
            f"F1={aggregate['mean_macro_f1']:.4f}±{aggregate['std_macro_f1']:.4f}"
        )

        return aggregate

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
