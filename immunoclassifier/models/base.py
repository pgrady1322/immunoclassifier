"""
Abstract base class for all immune cell classifiers.

Defines the interface that all model implementations must follow,
ensuring consistent training, prediction, evaluation, and serialization.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import anndata as ad

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """
    Abstract base classifier for immune cell types.

    All model implementations (logistic, XGBoost, GNN, scVI, foundation)
    must subclass this and implement the abstract methods.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.is_trained = False
        self.classes_ = None
        self.config = kwargs
        self._metadata = {}

    @abstractmethod
    def train(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        val_fraction: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the classifier on labeled data.

        Parameters
        ----------
        adata
            AnnData object with preprocessed data
        label_key
            Column in adata.obs containing cell type labels
        val_fraction
            Fraction of data to use for validation
        **kwargs
            Model-specific training parameters

        Returns
        -------
        Dictionary with training metrics (loss, accuracy, etc.)
        """
        pass

    @abstractmethod
    def predict(
        self,
        adata: ad.AnnData,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Predict cell types for new data.

        Parameters
        ----------
        adata
            AnnData object with preprocessed data (same features as training)
        return_probabilities
            If True, return probability matrix instead of hard predictions

        Returns
        -------
        Array of predicted cell type labels (or probability matrix)
        """
        pass

    def predict_with_confidence(
        self,
        adata: ad.AnnData,
    ) -> tuple:
        """
        Predict cell types with confidence scores.

        Returns
        -------
        Tuple of (predictions, confidences) where confidences is the
        maximum probability for each cell.
        """
        probabilities = self.predict(adata, return_probabilities=True)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        confidences = np.max(probabilities, axis=1)
        return predictions, confidences

    @abstractmethod
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained model from disk."""
        pass

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        n_classes = len(self.classes_) if self.classes_ is not None else 0
        return f"{self.__class__.__name__}(name='{self.name}', status={status}, n_classes={n_classes})"
