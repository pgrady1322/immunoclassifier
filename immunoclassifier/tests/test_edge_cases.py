#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Tests for edge cases and data validation scenarios.

Author: Patrick Grady
License: MIT License - See LICENSE
"""

import os
import tempfile

import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix


class TestLogisticEdgeCases:
    """Edge-case tests for LogisticClassifier."""

    def test_two_class_problem(self):
        """Train on only 2 classes."""
        from immunoclassifier.models.logistic import LogisticClassifier

        np.random.seed(42)
        X = csr_matrix(np.random.rand(100, 50).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G_{i}" for i in range(50)]
        adata.obs["cell_type"] = np.random.choice(["A", "B"], 100)
        adata.layers["counts"] = X.copy()

        model = LogisticClassifier()
        metrics = model.train(adata, label_key="cell_type")
        assert len(model.classes_) == 2
        assert model.is_trained

    def test_dense_matrix_input(self):
        """Train with dense matrix instead of sparse."""
        from immunoclassifier.models.logistic import LogisticClassifier

        np.random.seed(42)
        X = np.random.rand(80, 40).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G_{i}" for i in range(40)]
        adata.obs["cell_type"] = np.random.choice(["A", "B", "C"], 80)
        adata.layers["counts"] = X.copy()

        model = LogisticClassifier()
        metrics = model.train(adata, label_key="cell_type")
        assert model.is_trained
        preds = model.predict(adata)
        assert len(preds) == 80

    def test_custom_label_key(self, mock_adata):
        """Train with a non-default label column."""
        from immunoclassifier.models.logistic import LogisticClassifier

        mock_adata.obs["custom_labels"] = mock_adata.obs["cell_type"].copy()
        model = LogisticClassifier()
        metrics = model.train(mock_adata, label_key="custom_labels")
        assert model.is_trained


class TestXGBoostEdgeCases:
    """Edge-case tests for XGBoostClassifier."""

    def test_predict_untrained_raises(self, mock_adata):
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        model = XGBoostClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(mock_adata)

    def test_many_classes(self):
        """Train with many cell types."""
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        np.random.seed(42)
        n_cells, n_genes = 300, 80
        X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))
        adata = ad.AnnData(X=X)
        adata.var_names = [f"G_{i}" for i in range(n_genes)]
        types = [f"Type_{i}" for i in range(10)]
        adata.obs["cell_type"] = np.random.choice(types, n_cells)
        adata.layers["counts"] = X.copy()

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        metrics = model.train(adata, label_key="cell_type")
        assert len(model.classes_) == 10
        preds = model.predict(adata)
        assert all(p in types for p in preds)


class TestEvaluationEdgeCases:
    """Edge cases for evaluation metrics."""

    def test_single_class(self):
        from immunoclassifier.evaluation.metrics import evaluate_predictions

        y = np.array(["A", "A", "A"])
        results = evaluate_predictions(y, y)
        assert results["accuracy"] == 1.0
        assert results["n_classes_true"] == 1

    def test_mismatched_predictions(self):
        from immunoclassifier.evaluation.metrics import evaluate_predictions

        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["C", "C", "C", "C"])  # completely wrong, novel class
        results = evaluate_predictions(y_true, y_pred)
        assert results["accuracy"] == 0.0

    def test_per_class_all_correct(self):
        from immunoclassifier.evaluation.metrics import per_class_metrics

        y = np.array(["A", "B", "C", "A", "B", "C"])
        df = per_class_metrics(y, y)
        assert (df["precision"] == 1.0).all()
        assert (df["recall"] == 1.0).all()
        assert (df["f1"] == 1.0).all()

    def test_rare_cell_with_custom_threshold(self):
        from immunoclassifier.evaluation.metrics import rare_cell_analysis

        y_true = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 5)
        y_pred = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 5)

        # Threshold 10: C is rare
        results = rare_cell_analysis(y_true, y_pred, threshold=10)
        assert "C" in results["rare_types"]
        assert results["rare_accuracy"] == 1.0

        # Threshold 3: nothing is rare
        results2 = rare_cell_analysis(y_true, y_pred, threshold=3)
        assert len(results2["rare_types"]) == 0


class TestPreprocessingEdgeCases:
    """Edge cases for preprocessing."""

    def test_normalize_copy_true_returns_new_object(self, mock_adata):
        from immunoclassifier.data.preprocessing import normalize

        result = normalize(mock_adata, copy=True)
        assert result is not mock_adata
        assert result.raw is not None

    def test_normalize_preserves_shape(self, mock_adata):
        from immunoclassifier.data.preprocessing import normalize

        original_shape = mock_adata.shape
        result = normalize(mock_adata, copy=True)
        assert result.shape == original_shape


class TestDatasetsExtended:
    """Extended dataset registry tests."""

    def test_all_datasets_have_required_keys(self):
        from immunoclassifier.data.datasets import DATASETS

        required = {"description", "n_cells", "reference"}
        for name, entry in DATASETS.items():
            for key in required:
                assert key in entry, f"Dataset '{name}' missing key '{key}'"

    def test_immune_cell_hierarchy_is_nonempty(self):
        from immunoclassifier.data.datasets import IMMUNE_CELL_HIERARCHY

        assert len(IMMUNE_CELL_HIERARCHY) >= 5
        for parent, children in IMMUNE_CELL_HIERARCHY.items():
            assert isinstance(children, list)
            assert len(children) > 0

    def test_immune_markers_gene_lists(self):
        from immunoclassifier.data.datasets import IMMUNE_MARKERS

        assert len(IMMUNE_MARKERS) >= 10
        for cell_type, genes in IMMUNE_MARKERS.items():
            assert isinstance(genes, list)
            assert all(isinstance(g, str) for g in genes)


class TestConfigExtended:
    """Extended config tests."""

    def test_load_config_returns_dict(self):
        from immunoclassifier.utils.config import load_config

        config = load_config("configs/benchmark.yaml")
        assert isinstance(config, dict)

    def test_config_has_expected_structure(self):
        from immunoclassifier.utils.config import load_config

        config = load_config("configs/benchmark.yaml")
        assert "data" in config
        assert "models" in config
        # Check model entries
        models = config["models"]
        assert "logistic" in models
        assert "xgboost" in models


class TestFoundationModelExtended:
    """Extended foundation model tests."""

    def test_extract_embeddings_raises(self, mock_adata):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="scgpt")
        with pytest.raises(NotImplementedError):
            model.extract_embeddings(mock_adata)

    def test_save_raises(self, mock_adata):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="scgpt")
        with pytest.raises(NotImplementedError):
            model.save("/tmp/test_foundation")

    def test_load_raises(self):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="uce")
        with pytest.raises(NotImplementedError):
            model.load("/tmp/test_foundation")

    def test_all_backends_have_embedding_dim(self):
        from immunoclassifier.models.foundation import FOUNDATION_BACKENDS

        for name, info in FOUNDATION_BACKENDS.items():
            assert "embedding_dim" in info
            assert isinstance(info["embedding_dim"], int)
            assert info["embedding_dim"] > 0
