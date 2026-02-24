"""
Logistic Regression baseline classifier for immune cell types.

Simple but effective baseline using L2-regularized logistic regression
on highly variable genes. Surprisingly competitive for well-separated
cell types.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from immunoclassifier.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class LogisticClassifier(BaseClassifier):
    """
    L2-regularized logistic regression on HVG expression.

    This serves as the baseline method. Despite its simplicity,
    logistic regression performs well on immune cell classification
    when highly variable genes are informative.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, **kwargs):
        """
        Parameters
        ----------
        C
            Inverse regularization strength
        max_iter
            Maximum iterations for solver convergence
        """
        super().__init__(name="logistic_regression", **kwargs)
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        val_fraction: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train logistic regression on expression matrix."""
        logger.info(f"Training logistic regression (C={self.C})")

        X = self._get_features(adata)
        y = self.label_encoder.fit_transform(adata.obs[label_key].values)
        self.classes_ = self.label_encoder.classes_

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=42
        )

        # Train
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "n_classes": len(self.classes_),
            "n_features": X.shape[1],
            "n_train": len(y_train),
            "n_val": len(y_val),
        }

        logger.info(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
        return metrics

    def predict(
        self,
        adata: ad.AnnData,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """Predict cell types."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X = self._get_features(adata)

        if return_probabilities:
            return self.model.predict_proba(X)
        else:
            y_pred = self.model.predict(X)
            return self.label_encoder.inverse_transform(y_pred)

    def _get_features(self, adata: ad.AnnData) -> np.ndarray:
        """Extract feature matrix from AnnData."""
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        return np.asarray(X)

    def save(self, path: str) -> None:
        """Save model to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "classes_": self.classes_,
                    "config": {"C": self.C, "max_iter": self.max_iter},
                },
                f,
            )
        logger.info(f"Saved logistic classifier to {path}")

    def load(self, path: str) -> None:
        """Load model from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.classes_ = data["classes_"]
        self.is_trained = True
        logger.info(f"Loaded logistic classifier from {path}")
