"""
Tests for ImmunoClassifier.
"""

import pytest
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix


@pytest.fixture
def mock_adata():
    """Create a small mock AnnData for testing."""
    np.random.seed(42)
    n_cells = 200
    n_genes = 100
    n_types = 4

    # Random expression data
    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))

    # Gene names
    var_names = [f"Gene_{i}" for i in range(n_genes)]

    # Cell type labels
    cell_types = ["CD4+ T cell", "CD8+ T cell", "B cell", "Monocyte"]
    labels = np.random.choice(cell_types, size=n_cells)

    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs["cell_type"] = labels
    adata.layers["counts"] = X.copy()

    return adata


class TestLogisticClassifier:
    """Test logistic regression classifier."""

    def test_train_predict(self, mock_adata):
        from immunoclassifier.models.logistic import LogisticClassifier

        model = LogisticClassifier(C=1.0)
        metrics = model.train(mock_adata, label_key="cell_type")

        assert model.is_trained
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert len(model.classes_) == 4

        # Predict
        predictions = model.predict(mock_adata)
        assert len(predictions) == mock_adata.n_obs
        assert all(p in model.classes_ for p in predictions)

    def test_predict_probabilities(self, mock_adata):
        from immunoclassifier.models.logistic import LogisticClassifier

        model = LogisticClassifier()
        model.train(mock_adata, label_key="cell_type")

        probs = model.predict(mock_adata, return_probabilities=True)
        assert probs.shape == (mock_adata.n_obs, 4)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestXGBoostClassifier:
    """Test XGBoost classifier."""

    def test_train_predict(self, mock_adata):
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        metrics = model.train(mock_adata, label_key="cell_type")

        assert model.is_trained
        assert "train_accuracy" in metrics

        predictions = model.predict(mock_adata)
        assert len(predictions) == mock_adata.n_obs

    def test_feature_importance(self, mock_adata):
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.train(mock_adata, label_key="cell_type")

        importance = model.get_feature_importance(top_n=10)
        assert isinstance(importance, dict)
        assert len(importance) <= 10


class TestEvaluation:
    """Test evaluation metrics."""

    def test_evaluate_predictions(self):
        from immunoclassifier.evaluation.metrics import evaluate_predictions

        y_true = np.array(["A", "A", "B", "B", "C", "C"])
        y_pred = np.array(["A", "A", "B", "C", "C", "C"])

        results = evaluate_predictions(y_true, y_pred)
        assert "accuracy" in results
        assert "balanced_accuracy" in results
        assert "macro_f1" in results
        assert results["accuracy"] == pytest.approx(5 / 6)

    def test_rare_cell_analysis(self):
        from immunoclassifier.evaluation.metrics import rare_cell_analysis

        y_true = np.array(["A"] * 200 + ["B"] * 50 + ["C"] * 5)
        y_pred = np.array(["A"] * 200 + ["B"] * 50 + ["A"] * 5)  # C always misclassified

        results = rare_cell_analysis(y_true, y_pred, threshold=100)
        assert "C" in results["rare_types"]
        assert "A" in results["abundant_types"]
        assert results["rare_accuracy"] == 0.0  # All C predicted as A


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_normalize(self, mock_adata):
        from immunoclassifier.data.preprocessing import normalize

        adata = normalize(mock_adata, copy=True)
        assert adata.raw is not None
