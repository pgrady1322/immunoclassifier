#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Comprehensive test suite — models, evaluation, preprocessing, CLI, persistence.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import os
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# ── Fixtures ─────────────────────────────────────────────────────────


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


# ── Package Metadata ─────────────────────────────────────────────────


class TestPackageMetadata:
    """Test package-level imports and metadata."""

    def test_version(self):
        from immunoclassifier import __version__
        assert __version__ == "0.1.0"

    def test_top_level_imports(self):
        from immunoclassifier import data, evaluation, models
        assert data is not None
        assert models is not None
        assert evaluation is not None

    def test_model_registry_keys(self):
        from immunoclassifier.training.trainer import MODEL_REGISTRY

        # logistic and xgboost are always available
        assert "logistic" in MODEL_REGISTRY
        assert "xgboost" in MODEL_REGISTRY
        # scvi and gnn are optional (require torch)
        try:
            import torch  # noqa: F401

            assert "scvi" in MODEL_REGISTRY
            assert "gnn" in MODEL_REGISTRY
        except ImportError:
            pass


# ── Logistic Classifier ─────────────────────────────────────────────


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

    def test_predict_untrained_raises(self, mock_adata):
        from immunoclassifier.models.logistic import LogisticClassifier

        model = LogisticClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(mock_adata)

    def test_predict_with_confidence(self, mock_adata, trained_logistic):
        preds, confs = trained_logistic.predict_with_confidence(mock_adata)
        assert len(preds) == mock_adata.n_obs
        assert len(confs) == mock_adata.n_obs
        assert all(0 <= c <= 1 for c in confs)
        assert all(p in trained_logistic.classes_ for p in preds)

    def test_save_load_roundtrip(self, mock_adata, trained_logistic):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "logistic.pkl")
            trained_logistic.save(path)
            assert os.path.exists(path)

            from immunoclassifier.models.logistic import LogisticClassifier

            loaded = LogisticClassifier()
            loaded.load(path)

            assert loaded.is_trained
            assert np.array_equal(loaded.classes_, trained_logistic.classes_)

            y_orig = trained_logistic.predict(mock_adata)
            y_loaded = loaded.predict(mock_adata)
            assert np.array_equal(y_orig, y_loaded)

    def test_repr(self, trained_logistic):
        r = repr(trained_logistic)
        assert "trained" in r
        assert "LogisticClassifier" in r


# ── XGBoost Classifier ──────────────────────────────────────────────


class TestXGBoostClassifier:
    """Test XGBoost classifier."""

    def test_train_predict(self, mock_adata):
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        metrics = model.train(mock_adata, label_key="cell_type")

        assert model.is_trained
        assert "train_accuracy" in metrics
        assert "best_iteration" in metrics

        predictions = model.predict(mock_adata)
        assert len(predictions) == mock_adata.n_obs

    def test_predict_probabilities(self, mock_adata, trained_xgb):
        probs = trained_xgb.predict(mock_adata, return_probabilities=True)
        assert probs.shape == (mock_adata.n_obs, 4)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance(self, mock_adata, trained_xgb):
        importance = trained_xgb.get_feature_importance(top_n=10)
        assert isinstance(importance, dict)
        assert len(importance) <= 10

    def test_feature_importance_untrained_raises(self):
        from immunoclassifier.models.xgboost_model import XGBoostClassifier

        model = XGBoostClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.get_feature_importance()

    def test_save_load_roundtrip(self, mock_adata, trained_xgb):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "xgb_model")
            trained_xgb.save(path)

            from immunoclassifier.models.xgboost_model import XGBoostClassifier

            loaded = XGBoostClassifier()
            loaded.load(path)

            assert loaded.is_trained
            y_orig = trained_xgb.predict(mock_adata)
            y_loaded = loaded.predict(mock_adata)
            assert np.array_equal(y_orig, y_loaded)


# ── Evaluation Metrics ───────────────────────────────────────────────


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
        assert "per_class" in results
        assert "report" in results
        assert results["accuracy"] == pytest.approx(5 / 6)
        assert results["n_cells"] == 6
        assert results["n_classes_true"] == 3

    def test_per_class_metrics(self):
        from immunoclassifier.evaluation.metrics import per_class_metrics

        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "B", "B", "B"])

        df = per_class_metrics(y_true, y_pred)
        assert isinstance(df, pd.DataFrame)
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert df.loc["B", "recall"] == pytest.approx(1.0)

    def test_rare_cell_analysis(self):
        from immunoclassifier.evaluation.metrics import rare_cell_analysis

        y_true = np.array(["A"] * 200 + ["B"] * 150 + ["C"] * 5)
        y_pred = np.array(["A"] * 200 + ["B"] * 150 + ["A"] * 5)

        results = rare_cell_analysis(y_true, y_pred, threshold=100)
        assert "C" in results["rare_types"]
        assert "A" in results["abundant_types"]
        assert "B" in results["abundant_types"]
        assert results["rare_accuracy"] == 0.0  # all C predicted as A
        assert results["rare_n_cells"] == 5
        assert results["threshold"] == 100

    def test_rare_cell_no_rare_types(self):
        from immunoclassifier.evaluation.metrics import rare_cell_analysis

        y_true = np.array(["A"] * 200 + ["B"] * 200)
        y_pred = y_true.copy()

        results = rare_cell_analysis(y_true, y_pred, threshold=100)
        assert len(results["rare_types"]) == 0
        assert results["abundant_accuracy"] == 1.0

    def test_perfect_predictions(self):
        from immunoclassifier.evaluation.metrics import evaluate_predictions

        y = np.array(["A", "B", "C"] * 10)
        results = evaluate_predictions(y, y)
        assert results["accuracy"] == 1.0
        assert results["cohen_kappa"] == 1.0


# ── Preprocessing ────────────────────────────────────────────────────


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_normalize(self, mock_adata):
        from immunoclassifier.data.preprocessing import normalize

        adata = normalize(mock_adata, copy=True)
        assert adata.raw is not None

    def test_normalize_inplace(self, mock_adata):
        from immunoclassifier.data.preprocessing import normalize

        adata_copy = mock_adata.copy()
        result = normalize(adata_copy, copy=False)
        assert result.raw is not None
        # In-place should modify the same object
        assert result is adata_copy

    def test_select_hvgs(self, mock_adata):
        pytest.importorskip("skmisc", reason="skmisc required for seurat_v3 HVGs")
        from immunoclassifier.data.preprocessing import normalize, select_hvgs

        adata = normalize(mock_adata, copy=True)
        adata = select_hvgs(adata, n_top_genes=50)
        assert adata.n_vars <= 50
        assert "highly_variable" not in adata.var.columns or adata.var["highly_variable"].all()

    def test_full_preprocess_pipeline(self, mock_adata):
        pytest.importorskip("skmisc", reason="skmisc required for seurat_v3 HVGs")
        from immunoclassifier.data.preprocessing import preprocess

        mock_adata.var_names = [f"Gene_{i}" for i in range(mock_adata.n_vars)]

        adata = preprocess(
            mock_adata,
            min_genes=5,
            min_cells=1,
            max_pct_mito=100,
            n_top_genes=50,
            n_pcs=10,
            copy=True,
        )
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm
        assert "leiden" in adata.obs.columns
        assert adata.n_vars <= 50


# ── Datasets Registry ───────────────────────────────────────────────


class TestDatasets:
    """Test dataset registry and metadata."""

    def test_list_available_datasets(self):
        from immunoclassifier.data.datasets import list_available_datasets

        datasets = list_available_datasets()
        assert "pbmc_10k" in datasets
        assert "tabula_sapiens_immune" in datasets
        assert "hao_cite_seq" in datasets

    def test_immune_markers_registry(self):
        from immunoclassifier.data.datasets import IMMUNE_MARKERS

        assert "CD4+ T cells" in IMMUNE_MARKERS
        assert "CD3E" in IMMUNE_MARKERS["CD4+ T cells"]
        assert "pDC" in IMMUNE_MARKERS
        assert "NK cells" in IMMUNE_MARKERS

    def test_immune_cell_hierarchy(self):
        from immunoclassifier.data.datasets import IMMUNE_CELL_HIERARCHY

        assert "CD4+ T cells" in IMMUNE_CELL_HIERARCHY
        assert "Treg" in IMMUNE_CELL_HIERARCHY["CD4+ T cells"]
        assert "B cells" in IMMUNE_CELL_HIERARCHY
        assert "Plasma cell" in IMMUNE_CELL_HIERARCHY["B cells"]

    def test_dataset_metadata_fields(self):
        from immunoclassifier.data.datasets import DATASETS

        for name, info in DATASETS.items():
            assert "description" in info, f"{name} missing 'description'"
            assert "n_cells" in info, f"{name} missing 'n_cells'"
            assert "reference" in info, f"{name} missing 'reference'"


# ── Config Loader ────────────────────────────────────────────────────


class TestConfig:
    """Test YAML config loading."""

    def test_load_config(self):
        from immunoclassifier.utils.config import load_config

        # Load the bundled benchmark config
        config = load_config("configs/benchmark.yaml")
        assert "data" in config
        assert "models" in config
        assert "preprocessing" in config
        assert config["models"]["logistic"]["enabled"] is True

    def test_load_config_missing_file(self):
        from immunoclassifier.utils.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    """Test Click CLI commands."""

    def test_main_help(self):
        from click.testing import CliRunner

        from immunoclassifier.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ImmunoClassifier" in result.output

    def test_version_flag(self):
        from click.testing import CliRunner

        from immunoclassifier.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_download_unknown_dataset(self):
        from click.testing import CliRunner

        from immunoclassifier.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["download", "-d", "nonexistent"])
        assert result.exit_code == 0
        assert "Unknown dataset" in result.output

    def test_subcommands_registered(self):
        from immunoclassifier.cli import main

        command_names = list(main.commands.keys())
        assert "download" in command_names
        assert "train" in command_names
        assert "predict" in command_names
        assert "benchmark" in command_names


# ── Plots (smoke tests — verify they return Figure without error) ────


class TestPlots:
    """Smoke tests for plotting functions."""

    def test_plot_confusion_matrix(self):
        import matplotlib
        matplotlib.use("Agg")
        from immunoclassifier.evaluation.plots import plot_confusion_matrix

        y_true = np.array(["A", "A", "B", "B", "C"])
        y_pred = np.array(["A", "B", "B", "C", "C"])

        fig = plot_confusion_matrix(y_true, y_pred, normalize=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_confusion_matrix_unnormalized(self):
        import matplotlib
        matplotlib.use("Agg")
        from immunoclassifier.evaluation.plots import plot_confusion_matrix

        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "A", "B", "B"])

        fig = plot_confusion_matrix(y_true, y_pred, normalize=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_benchmark_comparison(self):
        import matplotlib
        matplotlib.use("Agg")
        from immunoclassifier.evaluation.plots import plot_benchmark_comparison

        results = {
            "ModelA": {"accuracy": 0.9, "macro_f1": 0.85},
            "ModelB": {"accuracy": 0.8, "macro_f1": 0.75},
        }
        fig = plot_benchmark_comparison(results, metrics=["accuracy", "macro_f1"])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# ── Foundation Model (stub behaviour) ────────────────────────────────


class TestFoundationModel:
    """Test foundation model stub raises NotImplementedError."""

    def test_train_raises(self, mock_adata):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="scgpt")
        with pytest.raises(NotImplementedError, match="planned for v0.2"):
            model.train(mock_adata)

    def test_predict_raises(self, mock_adata):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="geneformer")
        with pytest.raises(NotImplementedError):
            model.predict(mock_adata)

    def test_supported_backends(self):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        for backend in ("scgpt", "geneformer", "uce"):
            model = FoundationModelClassifier(backend=backend)
            assert model.name == f"foundation_{backend}"

    def test_invalid_backend_raises(self):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        with pytest.raises(ValueError, match="Unknown backend"):
            FoundationModelClassifier(backend="invalid")

    def test_list_backends(self):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        backends = FoundationModelClassifier.list_backends()
        assert set(backends.keys()) == {"scgpt", "geneformer", "uce"}
        for info in backends.values():
            assert "embedding_dim" in info
            assert "reference" in info

    def test_embedding_dim_set(self):
        from immunoclassifier.models.foundation import FoundationModelClassifier

        model = FoundationModelClassifier(backend="scgpt")
        assert model.embedding_dim == 512
        model_gf = FoundationModelClassifier(backend="geneformer")
        assert model_gf.embedding_dim == 256


# ── Base Classifier Interface ────────────────────────────────────────


class TestBaseClassifier:
    """Test abstract base classifier contract."""

    def test_cannot_instantiate_base(self):
        from immunoclassifier.models.base import BaseClassifier

        with pytest.raises(TypeError):
            BaseClassifier(name="test")

    def test_repr_untrained(self):
        from immunoclassifier.models.logistic import LogisticClassifier

        model = LogisticClassifier()
        r = repr(model)
        assert "untrained" in r
        assert "n_classes=0" in r


# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
