"""
XGBoost classifier for immune cell types.

Gradient-boosted trees on marker gene expression features.
Strong performance on tabular biological data with built-in
feature importance for biological interpretability.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from immunoclassifier.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class XGBoostClassifier(BaseClassifier):
    """
    XGBoost gradient-boosted tree classifier for immune cell types.

    Leverages XGBoost's handling of:
    - High-dimensional sparse features (gene expression)
    - Class imbalance (rare immune subtypes)
    - Feature importance (biologically interpretable)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_estimators
            Number of boosting rounds
        max_depth
            Maximum tree depth
        learning_rate
            Step size shrinkage
        subsample
            Subsample ratio of training instances
        colsample_bytree
            Subsample ratio of features per tree
        min_child_weight
            Minimum sum of instance weight in a child
        use_gpu
            Whether to use GPU acceleration
        """
        super().__init__(name="xgboost", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.use_gpu = use_gpu
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def train(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        val_fraction: float = 0.1,
        early_stopping_rounds: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train XGBoost classifier."""
        import xgboost as xgb

        logger.info(f"Training XGBoost (n_estimators={self.n_estimators}, max_depth={self.max_depth})")

        X = self._get_features(adata)
        y = self.label_encoder.fit_transform(adata.obs[label_key].values)
        self.classes_ = self.label_encoder.classes_
        self.feature_names = list(adata.var_names)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=42
        )

        # Compute class weights for imbalanced immune subtypes
        class_counts = np.bincount(y_train)
        n_samples = len(y_train)
        sample_weights = np.array([n_samples / (len(class_counts) * class_counts[yi]) for yi in y_train])

        # Configure XGBoost
        params = {
            "objective": "multi:softprob",
            "num_class": len(self.classes_),
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "eval_metric": ["mlogloss", "merror"],
            "seed": 42,
            "n_jobs": -1,
        }

        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        self.is_trained = True

        # Evaluate
        y_pred_train = np.argmax(self.model.predict(dtrain), axis=1)
        y_pred_val = np.argmax(self.model.predict(dval), axis=1)
        train_acc = np.mean(y_pred_train == y_train)
        val_acc = np.mean(y_pred_val == y_val)

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "n_classes": len(self.classes_),
            "n_features": X.shape[1],
            "n_train": len(y_train),
            "n_val": len(y_val),
            "best_iteration": self.model.best_iteration,
            "evals_result": evals_result,
        }

        logger.info(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
        return metrics

    def predict(
        self,
        adata: ad.AnnData,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """Predict cell types."""
        import xgboost as xgb

        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = self._get_features(adata)
        dmat = xgb.DMatrix(X)
        probabilities = self.model.predict(dmat)

        if return_probabilities:
            return probabilities
        else:
            y_pred = np.argmax(probabilities, axis=1)
            return self.label_encoder.inverse_transform(y_pred)

    def get_feature_importance(self, importance_type: str = "gain", top_n: int = 30) -> Dict[str, float]:
        """
        Get top feature importances (gene importances).

        Parameters
        ----------
        importance_type
            Type of importance: 'gain', 'weight', or 'cover'
        top_n
            Number of top features to return

        Returns
        -------
        Dictionary mapping gene names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")

        scores = self.model.get_score(importance_type=importance_type)

        # Map feature indices back to gene names
        gene_importance = {}
        for feat_key, score in scores.items():
            idx = int(feat_key.replace("f", ""))
            if self.feature_names and idx < len(self.feature_names):
                gene_importance[self.feature_names[idx]] = score
            else:
                gene_importance[feat_key] = score

        # Sort by importance
        sorted_importance = dict(
            sorted(gene_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        return sorted_importance

    def _get_features(self, adata: ad.AnnData) -> np.ndarray:
        """Extract feature matrix from AnnData."""
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X)

    def save(self, path: str) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path.with_suffix(".xgb")))

        # Save metadata
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump(
                {
                    "label_encoder": self.label_encoder,
                    "classes_": self.classes_,
                    "feature_names": self.feature_names,
                    "config": self.config,
                },
                f,
            )
        logger.info(f"Saved XGBoost classifier to {path}")

    def load(self, path: str) -> None:
        """Load model."""
        import xgboost as xgb

        path = Path(path)

        self.model = xgb.Booster()
        self.model.load_model(str(path.with_suffix(".xgb")))

        with open(path.with_suffix(".meta"), "rb") as f:
            meta = pickle.load(f)
        self.label_encoder = meta["label_encoder"]
        self.classes_ = meta["classes_"]
        self.feature_names = meta["feature_names"]
        self.is_trained = True
        logger.info(f"Loaded XGBoost classifier from {path}")
