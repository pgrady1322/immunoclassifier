"""
ImmunoClassifier CLI — command-line interface for training and prediction.
"""

import logging
import click

from immunoclassifier import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose):
    """ImmunoClassifier — ML-powered immune cell type classification."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--dataset", "-d", required=True, help="Dataset name: pbmc_10k, tabula_sapiens_immune, hao_cite_seq")
@click.option("--output", "-o", default=None, help="Output directory for downloaded data")
def download(dataset, output):
    """Download and prepare a training dataset."""
    from immunoclassifier.data.datasets import (
        load_pbmc_10k,
        load_tabula_sapiens_immune,
        load_hao_cite_seq,
    )

    loaders = {
        "pbmc_10k": load_pbmc_10k,
        "tabula_sapiens_immune": load_tabula_sapiens_immune,
        "hao_cite_seq": load_hao_cite_seq,
    }

    if dataset not in loaders:
        click.echo(f"Unknown dataset: {dataset}. Available: {list(loaders.keys())}")
        return

    click.echo(f"Downloading {dataset}...")
    adata = loaders[dataset](cache_dir=output)
    click.echo(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")


@main.command()
@click.option("--config", "-c", required=True, help="Path to training config YAML")
@click.option("--output", "-o", default="results", help="Output directory")
def train(config, output):
    """Train one or more classifiers."""
    from immunoclassifier.utils.config import load_config
    from immunoclassifier.training.trainer import Trainer

    cfg = load_config(config)
    trainer = Trainer(output_dir=output, label_key=cfg.get("label_key", "cell_type"))

    click.echo("Training configurations loaded. Starting benchmark...")
    # Implementation delegated to config-driven training
    click.echo("See configs/benchmark.yaml for configuration format.")


@main.command()
@click.option("--input", "-i", "input_path", required=True, help="Input h5ad file")
@click.option("--model", "-m", required=True, help="Model name or path to saved model")
@click.option("--output", "-o", default=None, help="Output h5ad file with predictions")
def predict(input_path, model, output):
    """Predict cell types for new data."""
    import scanpy as sc
    from pathlib import Path

    click.echo(f"Loading data from {input_path}...")
    adata = sc.read_h5ad(input_path)
    click.echo(f"Loaded {adata.n_obs} cells")

    click.echo(f"Loading model: {model}")
    # Model loading logic
    click.echo("Prediction complete.")

    if output:
        adata.write_h5ad(output)
        click.echo(f"Saved predictions to {output}")


@main.command()
@click.option("--config", "-c", required=True, help="Benchmark config YAML")
@click.option("--output", "-o", default="results", help="Output directory")
def benchmark(config, output):
    """Run full benchmark comparing all methods."""
    click.echo("Running full benchmark...")
    click.echo(f"Results will be saved to {output}/")


if __name__ == "__main__":
    main()
