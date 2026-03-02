#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Tests for hyperparameter optimization (hyperopt module).

Author: Patrick Grady
License: MIT License - See LICENSE
"""

import numpy as np
import pytest


class TestDefaultSearchSpaces:
    """Test DEFAULT_SEARCH_SPACES structure."""

    def test_logistic_space_exists(self):
        from immunoclassifier.training.hyperopt import DEFAULT_SEARCH_SPACES

        assert "logistic" in DEFAULT_SEARCH_SPACES
        space = DEFAULT_SEARCH_SPACES["logistic"]
        assert "C" in space
        assert "max_iter" in space

    def test_xgboost_space_exists(self):
        from immunoclassifier.training.hyperopt import DEFAULT_SEARCH_SPACES

        assert "xgboost" in DEFAULT_SEARCH_SPACES
        space = DEFAULT_SEARCH_SPACES["xgboost"]
        assert "n_estimators" in space
        assert "max_depth" in space
        assert "learning_rate" in space

    def test_gnn_space_exists(self):
        from immunoclassifier.training.hyperopt import DEFAULT_SEARCH_SPACES

        assert "gnn" in DEFAULT_SEARCH_SPACES
        space = DEFAULT_SEARCH_SPACES["gnn"]
        assert "hidden_channels" in space
        assert "n_heads" in space

    def test_scvi_space_exists(self):
        from immunoclassifier.training.hyperopt import DEFAULT_SEARCH_SPACES

        assert "scvi" in DEFAULT_SEARCH_SPACES
        space = DEFAULT_SEARCH_SPACES["scvi"]
        assert "n_latent" in space
        assert "n_hidden" in space

    def test_each_param_has_type(self):
        from immunoclassifier.training.hyperopt import DEFAULT_SEARCH_SPACES

        for model_name, space in DEFAULT_SEARCH_SPACES.items():
            for param_name, spec in space.items():
                assert "type" in spec, f"{model_name}.{param_name} missing 'type'"
                assert spec["type"] in {
                    "float",
                    "loguniform",
                    "int",
                    "categorical",
                }, f"{model_name}.{param_name} has unknown type '{spec['type']}'"


class TestSampleParam:
    """Test the _sample_param helper."""

    def test_sample_float(self):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import _sample_param

        study = optuna.create_study()
        trial = study.ask()
        val = _sample_param(trial, "test_f", {"type": "float", "low": 0.1, "high": 0.9})
        assert 0.1 <= val <= 0.9

    def test_sample_int(self):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import _sample_param

        study = optuna.create_study()
        trial = study.ask()
        val = _sample_param(trial, "test_i", {"type": "int", "low": 1, "high": 10})
        assert 1 <= val <= 10
        assert isinstance(val, int)

    def test_sample_loguniform(self):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import _sample_param

        study = optuna.create_study()
        trial = study.ask()
        val = _sample_param(trial, "test_lu", {"type": "loguniform", "low": 1e-4, "high": 1.0})
        assert 1e-4 <= val <= 1.0

    def test_sample_categorical(self):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import _sample_param

        study = optuna.create_study()
        trial = study.ask()
        val = _sample_param(
            trial, "test_c", {"type": "categorical", "choices": [32, 64, 128]}
        )
        assert val in [32, 64, 128]

    def test_sample_unknown_type_raises(self):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import _sample_param

        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown parameter type"):
            _sample_param(trial, "bad", {"type": "unknown"})


class TestTrainWithBestParams:
    """Test train_with_best_params."""

    def test_train_logistic_with_params(self, mock_adata):
        from immunoclassifier.training.hyperopt import train_with_best_params

        model, metrics = train_with_best_params(
            "logistic",
            mock_adata,
            best_params={"C": 0.5, "max_iter": 300},
        )
        assert model.is_trained
        assert "train_accuracy" in metrics
        assert model.C == 0.5

    def test_train_xgboost_with_params(self, mock_adata):
        from immunoclassifier.training.hyperopt import train_with_best_params

        model, metrics = train_with_best_params(
            "xgboost",
            mock_adata,
            best_params={"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
        )
        assert model.is_trained


class TestRunHyperopt:
    """Test run_hyperopt (requires optuna)."""

    def test_hyperopt_logistic_minimal(self, mock_adata):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import run_hyperopt

        result = run_hyperopt(
            "logistic",
            mock_adata,
            n_trials=3,
            n_folds=2,
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "best_trial_number" in result
        assert "metric" in result
        assert result["metric"] == "balanced_accuracy"
        assert "study_summary" in result
        assert result["n_trials"] == 3

    def test_hyperopt_unknown_model_raises(self, mock_adata):
        optuna = pytest.importorskip("optuna")
        from immunoclassifier.training.hyperopt import run_hyperopt

        with pytest.raises(ValueError, match="Unknown model"):
            run_hyperopt("nonexistent", mock_adata)

    def test_hyperopt_without_optuna(self, mock_adata, monkeypatch):
        """Verify ImportError is raised if optuna is not installed."""
        import importlib

        from immunoclassifier.training import hyperopt

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Reload to trigger the lazy import
        with pytest.raises(ImportError, match="Optuna is required"):
            # Call directly — optuna is imported lazily inside run_hyperopt
            hyperopt.run_hyperopt("logistic", mock_adata)

        monkeypatch.undo()
