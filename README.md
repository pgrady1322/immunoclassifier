# ImmunoClassifier

**ML-powered immune cell type classification from single-cell RNA-seq data.**

ImmunoClassifier benchmarks multiple machine learning approaches for fine-grained immune cell subtype identification, from classical ML baselines to deep learning and graph neural networks.

## End-to-End Demonstration with Figures

> **[PBMC Classification Benchmark Notebook](examples/pbmc_classification_benchmark.ipynb)**  
> Synthetic PBMC data → preprocessing → Logistic & XGBoost training → evaluation → visualizations.  
> No data downloads, no GPU — runs in ~30 seconds.

## Overview

Accurate immune cell type annotation is critical for interpreting scRNA-seq experiments in immunology, immuno-oncology, and infectious disease research. ImmunoClassifier provides:

- **Multiple ML architectures** benchmarked on the same data: logistic regression, XGBoost, scVI + neural classifier, GNN on cell-cell graphs, and foundation model embeddings
- **Fine-grained immune subtypes**: 20+ immune populations including T cell subsets (naive, memory, Th1/2/17, Treg, exhausted CD8+), B cell states, NK subsets, monocyte subtypes, and dendritic cell subsets
- **Systematic evaluation**: per-class metrics, confusion matrices, rare cell type performance, cross-dataset generalization
- **Simple prediction API**: `immunoclassifier predict --input adata.h5ad --model best`

## Immune Cell Types

| Compartment | Subtypes |
|---|---|
| CD4+ T cells | Naive, Central Memory, Effector Memory, Th1, Th2, Th17, Treg |
| CD8+ T cells | Naive, Effector, Memory, Exhausted |
| B cells | Naive, Memory, Plasma, Plasmablast |
| NK cells | CD56bright, CD56dim |
| Monocytes | Classical (CD14+), Non-classical (CD16+), Intermediate |
| Dendritic cells | cDC1, cDC2, pDC |
| Other | Macrophages, Platelets, HSPCs |

## Implemented Models

| Method | Category | Status |
|---|---|---|
| Logistic Regression | Baseline | Implemented — L2-regularized on HVGs |
| XGBoost | Classical ML | Implemented — gradient-boosted trees with class weighting |
| scVI + MLP | Deep Learning | Implemented — neural classifier on scVI latent space (requires `scvi-tools`) |
| Cell Graph GNN | Graph Neural Network | Implemented — GATv2Conv on cell-cell KNN graph (requires `torch-geometric`) |
| Foundation Models | Foundation Model | Planned — scGPT / Geneformer embedding interface scaffolded |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/immunoclassifier.git
cd immunoclassifier

# Create conda environment
conda env create -f env.yml
conda activate immunoclassifier

# Install package in development mode
pip install -e .
```

## Quick Start

```python
import immunoclassifier as ic

# Load and prepare data
adata = ic.data.load_pbmc_reference()
adata = ic.data.preprocess(adata)

# Train a model
model = ic.models.XGBoostClassifier()
model.train(adata, label_key="cell_type")

# Predict on new data
predictions = model.predict(new_adata)

# Evaluate
results = ic.evaluation.benchmark(adata_test, predictions, label_key="cell_type")
ic.evaluation.plot_confusion_matrix(results)
```

## CLI Usage

```bash
# Download and prepare training data
immunoclassifier download --dataset tabula-sapiens-immune

# Train all models
immunoclassifier train --config configs/benchmark.yaml

# Predict on new data
immunoclassifier predict --input my_data.h5ad --model xgboost --output predictions.h5ad

# Run full benchmark
immunoclassifier benchmark --config configs/benchmark.yaml --output results/
```

## Project Structure

```
immunoclassifier/
├── immunoclassifier/          # Main package
│   ├── __init__.py
│   ├── cli.py                 # Click CLI entry points
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py        # Public dataset downloaders
│   │   └── preprocessing.py   # QC, normalization, HVG selection
│   ├── models/                # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base classifier
│   │   ├── logistic.py        # Logistic regression baseline
│   │   ├── xgboost_model.py   # XGBoost classifier
│   │   ├── scvi_classifier.py # scVI latent + MLP
│   │   ├── gnn_classifier.py  # GATv2Conv on cell graph
│   │   └── foundation.py      # scGPT/Geneformer embeddings
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py         # Unified training loop
│   │   └── hyperopt.py        # Hyperparameter optimization
│   ├── evaluation/            # Metrics and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py         # Per-class metrics, rare cell analysis
│   │   └── plots.py           # Confusion matrices, UMAP overlays
│   └── utils/                 # Shared utilities
│       ├── __init__.py
│       └── config.py          # YAML config handling
├── configs/                   # Training configurations
│   └── benchmark.yaml
├── trained_models/            # Saved model weights
├── notebooks/                 # Analysis notebooks
│   └── 01_benchmark_analysis.ipynb
├── tests/                     # Unit tests
├── examples/                  # Example notebooks & scripts
│   └── pbmc_classification_benchmark.ipynb
├── env.yml                    # Conda environment
├── setup.py                   # Package installation
├── pyproject.toml             # Tool configuration
└── README.md
```

## Training Data

This package is designed to train on publicly available data sets:

- **Tabula Sapiens** (immune compartment) — ~100k cells, 40+ cell types
- **Hao et al. 2021 CITE-seq PBMC** — 161k PBMCs with protein-validated annotations
- **Human Cell Atlas Bone Marrow** — hematopoietic hierarchy

## Evaluation Strategy

1. **Within-dataset**: 5-fold cross-validation with stratified splits
2. **Cross-dataset**: Train on Tabula Sapiens, evaluate on Hao et al. PBMC
3. **Rare cell analysis**: Performance on cell types with <100 cells
4. **Uncertainty calibration**: Confidence score reliability
5. **Runtime comparison**: Training time, inference time, memory usage

## License

MIT License — see [LICENSE](LICENSE) for details.
