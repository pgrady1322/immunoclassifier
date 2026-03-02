#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

scVI latent space classifier with MLP cell-type head.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from immunoclassifier.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class CellTypeMLPHead(nn.Module):
    """MLP classification head on top of scVI latent space."""

    def __init__(self, n_latent: int, n_classes: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = n_latent
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ScVIClassifier(BaseClassifier):
    """
    scVI latent space + MLP classifier.

    Two-stage approach:
    1. Train scVI to learn a latent representation
    2. Train an MLP classifier on the latent coordinates
    """

    def __init__(
        self,
        n_latent: int = 30,
        n_layers: int = 2,
        n_hidden: int = 128,
        scvi_epochs: int = 100,
        classifier_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        batch_key: str | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_latent
            Dimensionality of the scVI latent space
        n_layers
            Number of hidden layers in scVI encoder/decoder
        n_hidden
            Hidden layer size in scVI
        scvi_epochs
            Training epochs for scVI
        classifier_epochs
            Training epochs for MLP classifier
        batch_size
            Batch size for training
        learning_rate
            Learning rate for MLP classifier
        batch_key
            Batch covariate column for scVI
        """
        super().__init__(name="scvi_classifier", **kwargs)
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.scvi_epochs = scvi_epochs
        self.classifier_epochs = classifier_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.batch_key = batch_key

        self.scvi_model = None
        self.classifier = None
        self.label_encoder = LabelEncoder()

    def train(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        val_fraction: float = 0.1,
        **kwargs,
    ) -> dict[str, Any]:
        """Train scVI + MLP classifier."""
        import scvi

        logger.info(f"Training scVI (latent={self.n_latent}, epochs={self.scvi_epochs})")

        # Stage 1: Train scVI
        # scVI needs raw counts
        adata_scvi = adata.copy()
        if "counts" in adata_scvi.layers:
            adata_scvi.X = adata_scvi.layers["counts"]

        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            layer=None,
            batch_key=self.batch_key,
        )

        self.scvi_model = scvi.model.SCVI(
            adata_scvi,
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
        )
        self.scvi_model.train(
            max_epochs=self.scvi_epochs,
            batch_size=self.batch_size,
            early_stopping=True,
        )

        # Get latent representation
        latent = self.scvi_model.get_latent_representation(adata_scvi)
        logger.info(f"scVI latent shape: {latent.shape}")

        # Stage 2: Train MLP classifier on latent space
        y = self.label_encoder.fit_transform(adata.obs[label_key].values)
        self.classes_ = self.label_encoder.classes_

        X_train, X_val, y_train, y_val = train_test_split(
            latent, y, test_size=val_fraction, stratify=y, random_state=42
        )

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")

        # Build classifier
        self.classifier = CellTypeMLPHead(
            n_latent=self.n_latent,
            n_classes=len(self.classes_),
        ).to(device)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.LongTensor(y_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.LongTensor(y_val).to(device)

        # Training loop
        best_val_acc = 0
        for epoch in range(self.classifier_epochs):
            self.classifier.train()

            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                optimizer.zero_grad()
                logits = self.classifier(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Validate
            self.classifier.eval()
            with torch.no_grad():
                val_logits = self.classifier(X_val_t)
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.classifier_epochs} — "
                    f"Loss: {epoch_loss/n_batches:.4f}, Val Acc: {val_acc:.4f}"
                )

        self.is_trained = True

        # Final metrics
        self.classifier.eval()
        with torch.no_grad():
            train_logits = self.classifier(X_train_t)
            train_acc = (train_logits.argmax(dim=1) == y_train_t).float().mean().item()

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": best_val_acc,
            "n_classes": len(self.classes_),
            "n_latent": self.n_latent,
            "n_train": len(y_train),
            "n_val": len(y_val),
        }

        logger.info(f"Train acc: {train_acc:.4f}, Best val acc: {best_val_acc:.4f}")
        return metrics

    def predict(
        self,
        adata: ad.AnnData,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """Predict cell types via scVI latent space + MLP."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        # Get latent representation
        adata_scvi = adata.copy()
        if "counts" in adata_scvi.layers:
            adata_scvi.X = adata_scvi.layers["counts"]

        latent = self.scvi_model.get_latent_representation(adata_scvi)

        # Predict
        device = next(self.classifier.parameters()).device
        X = torch.FloatTensor(latent).to(device)

        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        if return_probabilities:
            return probabilities
        else:
            y_pred = np.argmax(probabilities, axis=1)
            return self.label_encoder.inverse_transform(y_pred)

    def save(self, path: str) -> None:
        """Save scVI model + classifier."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save scVI model
        scvi_dir = path.parent / f"{path.stem}_scvi"
        self.scvi_model.save(str(scvi_dir), overwrite=True)

        # Save classifier + metadata
        torch.save(
            {
                "classifier_state": self.classifier.state_dict(),
                "label_encoder": self.label_encoder,
                "classes_": self.classes_,
                "n_latent": self.n_latent,
                "n_classes": len(self.classes_),
                "config": self.config,
            },
            str(path.with_suffix(".pt")),
        )
        logger.info(f"Saved scVI classifier to {path}")

    def load(self, path: str) -> None:
        """Load scVI model + classifier."""

        path = Path(path)

        # Load scVI
        path.parent / f"{path.stem}_scvi"
        # Note: loading scVI requires the original adata setup
        logger.warning("scVI model loading requires adata with same setup. Use scvi.model.SCVI.load()")

        # Load classifier
        checkpoint = torch.load(str(path.with_suffix(".pt")), map_location="cpu")
        self.label_encoder = checkpoint["label_encoder"]
        self.classes_ = checkpoint["classes_"]
        self.n_latent = checkpoint["n_latent"]

        self.classifier = CellTypeMLPHead(
            n_latent=checkpoint["n_latent"],
            n_classes=checkpoint["n_classes"],
        )
        self.classifier.load_state_dict(checkpoint["classifier_state"])
        self.is_trained = True
        logger.info(f"Loaded scVI classifier from {path}")

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
