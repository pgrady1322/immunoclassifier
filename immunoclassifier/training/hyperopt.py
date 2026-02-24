#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmunoClassifier v0.1.0

Optuna-based hyperparameter optimization for all classifier backends.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import anndata as ad

logger = logging.getLogger(__name__)

# Search spaces per model
DEFAULT_SEARCH_SPACES = {
    "logistic": {
        "C": {"type": "loguniform", "low": 1e-3, "high": 1e2},
        "max_iter": {"type": "int", "low": 200, "high": 2000, "step": 100},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 1000, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "loguniform", "low": 1e-3, "high": 0.3},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 20},
    },
    "gnn": {
        "hidden_channels": {"type": "categorical", "choices": [32, 64, 128]},
        "n_heads": {"type": "categorical", "choices": [2, 4, 8]},
        "n_layers": {"type": "int", "low": 2, "high": 5},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
        "epochs": {"type": "categorical", "choices": [100, 200, 300]},
    },
    "scvi": {
        "n_latent": {"type": "categorical", "choices": [10, 20, 30, 50]},
        "n_layers": {"type": "int", "low": 1, "high": 3},
        "n_hidden": {"type": "categorical", "choices": [64, 128, 256]},
        "scvi_epochs": {"type": "int", "low": 50, "high": 200, "step": 25},
        "classifier_epochs": {"type": "int", "low": 20, "high": 100, "step": 10},
    },
}


def _sample_param(trial, name: str, spec: Dict[str, Any]) -> Any:
    """Sample a single hyperparameter from an Optuna trial."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif ptype == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif ptype == "int":
        step = spec.get("step", 1)
        return trial.suggest_int(name, spec["low"], spec["high"], step=step)
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {ptype}")


def run_hyperopt(
    model_name: str,
    adata: ad.AnnData,
    label_key: str = "cell_type",
    metric: str = "balanced_accuracy",
    n_trials: int = 30,
    n_folds: int = 3,
    search_space: Optional[Dict[str, Dict]] = None,
    timeout: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for a given model.

    Uses stratified k-fold cross-validation as the objective, optimizing
    the specified metric.

    Parameters
    ----------
    model_name
        Classifier key: 'logistic', 'xgboost', 'gnn', or 'scvi'
    adata
        Preprocessed AnnData with cell type labels
    label_key
        Column in adata.obs containing labels
    metric
        Metric to optimize: 'accuracy', 'balanced_accuracy', or 'macro_f1'
    n_trials
        Number of Optuna trials
    n_folds
        Number of CV folds per trial
    search_space
        Custom search space dict. If None, uses DEFAULT_SEARCH_SPACES.
    timeout
        Maximum time in seconds for the study
    random_state
        Random seed

    Returns
    -------
    Dict with keys: best_params, best_score, study_summary
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install it with: pip install optuna"
        )

    from sklearn.model_selection import StratifiedKFold
    from immunoclassifier.training.trainer import MODEL_REGISTRY
    from immunoclassifier.evaluation.metrics import evaluate_predictions

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}"
        )

    space = search_space or DEFAULT_SEARCH_SPACES.get(model_name, {})
    if not space:
        raise ValueError(f"No search space defined for model '{model_name}'")

    y = adata.obs[label_key].values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    logger.info(
        f"Starting hyperopt: model={model_name}, metric={metric}, "
        f"trials={n_trials}, folds={n_folds}"
    )

    # Suppress Optuna's default logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        # Sample hyperparameters
        params = {name: _sample_param(trial, name, spec) for name, spec in space.items()}

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            adata_train = adata[train_idx].copy()
            adata_val = adata[val_idx].copy()

            try:
                model_cls = MODEL_REGISTRY[model_name]
                model = model_cls(**params)
                model.train(adata_train, label_key=label_key)

                y_pred = model.predict(adata_val)
                y_true_fold = adata_val.obs[label_key].values
                eval_result = evaluate_predictions(y_true_fold, y_pred)

                fold_scores.append(eval_result[metric])
            except Exception as e:
                logger.warning(f"Trial {trial.number}, fold {fold_idx} failed: {e}")
                return 0.0

        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    logger.info(
        f"Hyperopt complete: best {metric}={best.value:.4f} "
        f"(trial {best.number}/{n_trials})"
    )
    logger.info(f"Best params: {best.params}")

    # Build summary
    result = {
        "best_params": best.params,
        "best_score": best.value,
        "best_trial_number": best.number,
        "metric": metric,
        "n_trials": n_trials,
        "n_folds": n_folds,
        "study_summary": {
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        },
    }

    return result


def train_with_best_params(
    model_name: str,
    adata: ad.AnnData,
    best_params: Dict[str, Any],
    label_key: str = "cell_type",
) -> tuple:
    """
    Train a model using the best params from hyperopt.

    Parameters
    ----------
    model_name
        Classifier key
    adata
        Preprocessed AnnData
    best_params
        Best hyperparameters from run_hyperopt()
    label_key
        Label column

    Returns
    -------
    Tuple of (trained model, training metrics)
    """
    from immunoclassifier.training.trainer import MODEL_REGISTRY

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**best_params)
    metrics = model.train(adata, label_key=label_key)

    logger.info(f"Trained {model_name} with optimized params: val_acc={metrics.get('val_accuracy', 'N/A')}")
    return model, metrics

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
