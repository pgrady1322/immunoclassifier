#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

Tests for the Trainer class — train_model, benchmark, cross_validate.

Author: Patrick Grady
License: MIT License - See LICENSE
"""

import os
import tempfile

import numpy as np
import pytest


class TestTrainerInit:
    """Test Trainer constructor and output directory management."""

    def test_default_init(self, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path / "results"))
        assert trainer.label_key == "cell_type"
        assert trainer.random_state == 42
        assert trainer.results == {}
        assert trainer.output_dir.exists()

    def test_custom_init(self, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(
            output_dir=str(tmp_path / "custom_out"),
            label_key="celltype",
            random_state=123,
        )
        assert trainer.label_key == "celltype"
        assert trainer.random_state == 123
        assert trainer.output_dir.exists()

    def test_creates_nested_output_dir(self, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        deep_path = tmp_path / "a" / "b" / "c"
        trainer = Trainer(output_dir=str(deep_path))
        assert deep_path.exists()


class TestTrainerTrainModel:
    """Test Trainer.train_model()."""

    def test_train_logistic(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        model, metrics = trainer.train_model("logistic", mock_adata)

        assert model.is_trained
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "training_time_seconds" in metrics
        assert metrics["training_time_seconds"] >= 0

    def test_train_xgboost(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        model, metrics = trainer.train_model(
            "xgboost",
            mock_adata,
            model_kwargs={"n_estimators": 10, "max_depth": 3},
        )

        assert model.is_trained
        assert "training_time_seconds" in metrics

    def test_train_with_custom_kwargs(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        model, metrics = trainer.train_model(
            "logistic",
            mock_adata,
            model_kwargs={"C": 0.01, "max_iter": 500},
        )
        assert model.is_trained
        assert model.C == 0.01

    def test_train_unknown_model_raises(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model("nonexistent", mock_adata)

    def test_train_model_predictions_valid(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        model, _ = trainer.train_model("logistic", mock_adata)

        preds = model.predict(mock_adata)
        assert len(preds) == mock_adata.n_obs
        assert all(p in model.classes_ for p in preds)


class TestTrainerBenchmark:
    """Test Trainer.benchmark()."""

    def test_benchmark_logistic_only(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        results = trainer.benchmark(
            mock_adata,
            models=["logistic"],
        )

        assert "logistic" in results
        assert "training_metrics" in results["logistic"]

    def test_benchmark_with_test_set(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        # Split data
        train_idx = list(range(0, 150))
        test_idx = list(range(150, 200))
        adata_train = mock_adata[train_idx].copy()
        adata_test = mock_adata[test_idx].copy()

        trainer = Trainer(output_dir=str(tmp_path))
        results = trainer.benchmark(
            adata_train,
            adata_test=adata_test,
            models=["logistic"],
        )

        assert "logistic" in results
        assert "accuracy" in results["logistic"]
        assert "balanced_accuracy" in results["logistic"]
        assert "macro_f1" in results["logistic"]
        assert "training_metrics" in results["logistic"]
        assert "rare_cell_analysis" in results["logistic"]

    def test_benchmark_multiple_models(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        results = trainer.benchmark(
            mock_adata,
            models=["logistic", "xgboost"],
            model_configs={"xgboost": {"n_estimators": 10, "max_depth": 3}},
        )

        assert "logistic" in results
        assert "xgboost" in results

    def test_benchmark_saves_models(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        trainer.benchmark(mock_adata, models=["logistic"])

        model_dir = tmp_path / "models" / "logistic"
        assert model_dir.exists()

    def test_benchmark_stores_results(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        results = trainer.benchmark(mock_adata, models=["logistic"])

        # trainer.results should be set
        assert trainer.results == results

    def test_benchmark_default_models(self, mock_adata, tmp_path):
        """Benchmark with None models should use all available models."""
        from immunoclassifier.training.trainer import MODEL_REGISTRY, Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        results = trainer.benchmark(
            mock_adata,
            model_configs={"xgboost": {"n_estimators": 10, "max_depth": 3}},
        )

        # Should have tried every model in the registry
        for model_name in MODEL_REGISTRY:
            assert model_name in results


class TestTrainerCrossValidation:
    """Test Trainer.cross_validate()."""

    def test_cv_logistic(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        cv_results = trainer.cross_validate(
            "logistic",
            mock_adata,
            n_folds=3,
        )

        assert "mean_accuracy" in cv_results
        assert "std_accuracy" in cv_results
        assert "mean_balanced_accuracy" in cv_results
        assert "std_balanced_accuracy" in cv_results
        assert "mean_macro_f1" in cv_results
        assert "std_macro_f1" in cv_results
        assert cv_results["n_folds"] == 3
        assert len(cv_results["per_fold"]) == 3

    def test_cv_xgboost(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        cv_results = trainer.cross_validate(
            "xgboost",
            mock_adata,
            n_folds=2,
            model_kwargs={"n_estimators": 10, "max_depth": 3},
        )

        assert cv_results["n_folds"] == 2
        assert len(cv_results["per_fold"]) == 2
        assert 0 <= cv_results["mean_accuracy"] <= 1

    def test_cv_per_fold_has_metrics(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        cv_results = trainer.cross_validate("logistic", mock_adata, n_folds=2)

        for fold_result in cv_results["per_fold"]:
            assert "accuracy" in fold_result
            assert "balanced_accuracy" in fold_result
            assert "macro_f1" in fold_result

    def test_cv_aggregate_within_bounds(self, mock_adata, tmp_path):
        from immunoclassifier.training.trainer import Trainer

        trainer = Trainer(output_dir=str(tmp_path))
        cv_results = trainer.cross_validate("logistic", mock_adata, n_folds=3)

        assert 0 <= cv_results["mean_accuracy"] <= 1
        assert cv_results["std_accuracy"] >= 0
        assert 0 <= cv_results["mean_macro_f1"] <= 1


class TestModelRegistry:
    """Test MODEL_REGISTRY contents."""

    def test_registry_always_has_core_models(self):
        from immunoclassifier.training.trainer import MODEL_REGISTRY

        assert "logistic" in MODEL_REGISTRY
        assert "xgboost" in MODEL_REGISTRY

    def test_registry_values_are_classes(self):
        from immunoclassifier.training.trainer import MODEL_REGISTRY

        for name, cls in MODEL_REGISTRY.items():
            assert callable(cls), f"{name} should be a callable class"
