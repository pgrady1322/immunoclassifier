#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Shared fixtures for the test suite.

Author: Patrick Grady
License: MIT License - See LICENSE
"""

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def mock_adata():
    """Create a small mock AnnData for testing."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100

    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))
    var_names = [f"Gene_{i}" for i in range(n_genes)]

    cell_types = ["CD4+ T cell", "CD8+ T cell", "B cell", "Monocyte"]
    labels = np.random.choice(cell_types, size=n_cells)

    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs["cell_type"] = labels
    adata.layers["counts"] = X.copy()

    return adata


@pytest.fixture
def trained_logistic(mock_adata):
    """Return a trained LogisticClassifier."""
    from immunoclassifier.models.logistic import LogisticClassifier

    model = LogisticClassifier(C=1.0, max_iter=200)
    model.train(mock_adata, label_key="cell_type")
    return model


@pytest.fixture
def trained_xgb(mock_adata):
    """Return a trained XGBoostClassifier."""
    from immunoclassifier.models.xgboost_model import XGBoostClassifier

    model = XGBoostClassifier(n_estimators=10, max_depth=3)
    model.train(mock_adata, label_key="cell_type")
    return model
