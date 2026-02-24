#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmunoClassifier v0.1.0

Graph Neural Network classifier using GATv2Conv on cell KNN graphs.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from immunoclassifier.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class CellGraphGNN:
    """
    GATv2Conv-based GNN for cell type classification on cell-cell graphs.

    Requires torch and torch_geometric.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.3,
    ):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GATv2Conv

        self.device = self._get_device()

        # Build model
        class GATv2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()

                # Input layer
                self.convs.append(
                    GATv2Conv(n_features, hidden_channels, heads=n_heads, concat=True)
                )
                self.norms.append(nn.BatchNorm1d(hidden_channels * n_heads))

                # Hidden layers
                for _ in range(n_layers - 2):
                    self.convs.append(
                        GATv2Conv(
                            hidden_channels * n_heads,
                            hidden_channels,
                            heads=n_heads,
                            concat=True,
                        )
                    )
                    self.norms.append(nn.BatchNorm1d(hidden_channels * n_heads))

                # Output layer
                self.convs.append(
                    GATv2Conv(
                        hidden_channels * n_heads,
                        n_classes,
                        heads=1,
                        concat=False,
                    )
                )

                self.dropout = dropout

            def forward(self, x, edge_index):
                for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
                    x = conv(x, edge_index)
                    x = norm(x)
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                x = self.convs[-1](x, edge_index)
                return x

        self.model = GATv2Model().to(self.device)

    def _get_device(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class GNNClassifier(BaseClassifier):
    """
    GATv2Conv graph neural network for immune cell classification.

    Constructs a cell-cell KNN graph from the expression/PCA space,
    then runs graph attention convolution for node (cell) classification.

    This captures cell neighborhood structure — for example, transitional
    cell states between naive and memory T cells will be influenced by
    their graph neighbors, improving classification of ambiguous cells.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.3,
        n_neighbors: int = 15,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        use_pca: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        hidden_channels
            Hidden dimension per attention head
        n_heads
            Number of attention heads
        n_layers
            Number of GATv2Conv layers
        dropout
            Dropout rate
        n_neighbors
            Number of neighbors for KNN graph construction
        epochs
            Training epochs
        learning_rate
            Learning rate
        use_pca
            If True, use PCA coordinates as node features (otherwise HVG expression)
        """
        super().__init__(name="gnn_classifier", **kwargs)
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_neighbors = n_neighbors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.use_pca = use_pca

        self.gnn = None
        self.label_encoder = LabelEncoder()

    def _build_cell_graph(self, adata: ad.AnnData):
        """
        Build a cell-cell KNN graph as a PyTorch Geometric Data object.

        Uses scanpy's neighbor graph (computed during preprocessing)
        or builds one from PCA coordinates.
        """
        import torch
        from torch_geometric.data import Data
        from scipy.sparse import csr_matrix

        # Node features
        if self.use_pca and "X_pca" in adata.obsm:
            X = torch.FloatTensor(adata.obsm["X_pca"])
        else:
            import scipy.sparse as sp
            feat = adata.X
            if sp.issparse(feat):
                feat = feat.toarray()
            X = torch.FloatTensor(np.asarray(feat))

        # Edge index from neighbor graph
        if "connectivities" in adata.obsp:
            adj = adata.obsp["connectivities"]
        else:
            import scanpy as sc
            sc.pp.neighbors(adata, n_neighbors=self.n_neighbors)
            adj = adata.obsp["connectivities"]

        # Convert sparse adjacency to edge_index
        adj_coo = csr_matrix(adj).tocoo()
        edge_index = torch.LongTensor(np.array([adj_coo.row, adj_coo.col]))

        data = Data(x=X, edge_index=edge_index)
        logger.info(
            f"Cell graph: {data.num_nodes} nodes, {data.num_edges} edges, "
            f"{X.shape[1]} features"
        )
        return data

    def train(
        self,
        adata: ad.AnnData,
        label_key: str = "cell_type",
        val_fraction: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train GNN classifier on cell-cell graph."""
        import torch
        import torch.nn.functional as F

        logger.info(
            f"Training GNN (layers={self.n_layers}, heads={self.n_heads}, "
            f"hidden={self.hidden_channels})"
        )

        # Encode labels
        y = self.label_encoder.fit_transform(adata.obs[label_key].values)
        self.classes_ = self.label_encoder.classes_

        # Build graph
        graph_data = self._build_cell_graph(adata)
        n_features = graph_data.x.shape[1]

        # Create train/val masks
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_fraction, stratify=y, random_state=42
        )

        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        # Initialize GNN
        self.gnn = CellGraphGNN(
            n_features=n_features,
            n_classes=len(self.classes_),
            hidden_channels=self.hidden_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        device = self.gnn.device
        model = self.gnn.model

        # Move data to device
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        labels = torch.LongTensor(y).to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)

        # Training loop
        best_val_acc = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            logits = model(x, edge_index)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                train_acc = (logits[train_mask].argmax(dim=1) == labels[train_mask]).float().mean().item()
                val_acc = (logits[val_mask].argmax(dim=1) == labels[val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 25 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} — "
                    f"Loss: {loss.item():.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
                )

        self.is_trained = True

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": best_val_acc,
            "n_classes": len(self.classes_),
            "n_features": n_features,
            "n_nodes": graph_data.num_nodes,
            "n_edges": graph_data.num_edges,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        }

        logger.info(f"Train acc: {train_acc:.4f}, Best val acc: {best_val_acc:.4f}")
        return metrics

    def predict(
        self,
        adata: ad.AnnData,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """Predict cell types using trained GNN."""
        import torch

        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        graph_data = self._build_cell_graph(adata)
        device = self.gnn.device
        model = self.gnn.model

        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        if return_probabilities:
            return probabilities
        else:
            y_pred = np.argmax(probabilities, axis=1)
            return self.label_encoder.inverse_transform(y_pred)

    def save(self, path: str) -> None:
        """Save GNN model."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.gnn.model.state_dict(),
                "label_encoder": self.label_encoder,
                "classes_": self.classes_,
                "config": {
                    "hidden_channels": self.hidden_channels,
                    "n_heads": self.n_heads,
                    "n_layers": self.n_layers,
                    "dropout": self.dropout,
                },
            },
            str(path.with_suffix(".pt")),
        )
        logger.info(f"Saved GNN classifier to {path}")

    def load(self, path: str) -> None:
        """Load GNN model."""
        import torch

        path = Path(path)
        checkpoint = torch.load(str(path.with_suffix(".pt")), map_location="cpu")

        self.label_encoder = checkpoint["label_encoder"]
        self.classes_ = checkpoint["classes_"]
        self.is_trained = True

        logger.info(f"Loaded GNN classifier metadata from {path}")
        logger.warning("GNN model requires rebuild with correct input dimensions. Use train() or provide n_features.")

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
