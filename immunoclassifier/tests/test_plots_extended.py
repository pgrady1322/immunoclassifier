#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Tests for plots — UMAP predictions, save_path, edge cases.

Author: Patrick Grady
License: MIT License - See LICENSE
"""

import os

import anndata as ad
import matplotlib
import numpy as np
import pytest
from scipy.sparse import csr_matrix

matplotlib.use("Agg")


@pytest.fixture
def adata_with_umap():
    """Create AnnData with UMAP coordinates and predictions."""
    np.random.seed(42)
    n_cells = 50
    X = csr_matrix(np.random.rand(n_cells, 20).astype(np.float32))
    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], n_cells)
    adata.obs["predicted_cell_type"] = np.random.choice(["A", "B", "C"], n_cells)
    adata.obsm["X_umap"] = np.random.rand(n_cells, 2).astype(np.float32)
    return adata


class TestPlotUmapPredictions:
    """Test plot_umap_predictions."""

    def test_umap_with_true_and_predicted(self, adata_with_umap):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_umap_predictions

        fig = plot_umap_predictions(adata_with_umap)
        assert fig is not None
        plt.close(fig)

    def test_umap_predicted_only(self, adata_with_umap):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_umap_predictions

        fig = plot_umap_predictions(adata_with_umap, true_key=None)
        assert fig is not None
        plt.close(fig)

    def test_umap_missing_true_key(self, adata_with_umap):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_umap_predictions

        fig = plot_umap_predictions(adata_with_umap, true_key="nonexistent_col")
        assert fig is not None
        plt.close(fig)


class TestPlotConfusionMatrixSave:
    """Test confusion matrix save_path."""

    def test_save_confusion_matrix(self, tmp_path):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_confusion_matrix

        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "B", "B", "B"])

        path = str(tmp_path / "cm.png")
        fig = plot_confusion_matrix(y_true, y_pred, save_path=path)
        assert os.path.exists(path)
        plt.close(fig)

    def test_confusion_matrix_custom_labels(self):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_confusion_matrix

        y_true = np.array(["X", "X", "Y", "Y"])
        y_pred = np.array(["X", "Y", "Y", "Y"])

        fig = plot_confusion_matrix(y_true, y_pred, labels=["X", "Y"])
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_custom_title(self):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_confusion_matrix

        y_true = np.array(["A", "B"])
        y_pred = np.array(["A", "B"])

        fig = plot_confusion_matrix(y_true, y_pred, title="Custom Title")
        assert fig is not None
        plt.close(fig)


class TestPlotBenchmarkSave:
    """Test benchmark plot save_path."""

    def test_save_benchmark_plot(self, tmp_path):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_benchmark_comparison

        results = {
            "ModelA": {"accuracy": 0.9, "macro_f1": 0.85},
            "ModelB": {"accuracy": 0.8, "macro_f1": 0.75},
        }
        path = str(tmp_path / "bench.png")
        fig = plot_benchmark_comparison(results, save_path=path)
        assert os.path.exists(path)
        plt.close(fig)

    def test_benchmark_default_metrics(self):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_benchmark_comparison

        results = {
            "M1": {"accuracy": 0.9, "balanced_accuracy": 0.88, "macro_f1": 0.87},
        }
        fig = plot_benchmark_comparison(results)
        assert fig is not None
        plt.close(fig)

    def test_benchmark_empty_results(self):
        import matplotlib.pyplot as plt

        from immunoclassifier.evaluation.plots import plot_benchmark_comparison

        fig = plot_benchmark_comparison({})
        assert fig is not None
        plt.close(fig)


class TestXGBoostConfidence:
    """Test predict_with_confidence via XGBoost (only logistic was tested before)."""

    def test_xgb_predict_with_confidence(self, mock_adata, trained_xgb):
        preds, confs = trained_xgb.predict_with_confidence(mock_adata)
        assert len(preds) == mock_adata.n_obs
        assert len(confs) == mock_adata.n_obs
        assert all(0 <= c <= 1 for c in confs)
