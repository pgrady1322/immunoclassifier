"""
Preprocessing pipeline for immune cell classification.

Standardized QC, normalization, and feature selection
optimized for immune cell type classification tasks.
"""

import logging
from typing import Optional

import numpy as np
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


def preprocess(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mito: float = 20.0,
    n_top_genes: int = 3000,
    target_sum: float = 1e4,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    batch_key: Optional[str] = None,
    copy: bool = True,
) -> ad.AnnData:
    """
    End-to-end preprocessing pipeline for immune scRNA-seq data.

    Steps:
    1. QC filtering (cells and genes)
    2. Normalization and log-transformation
    3. HVG selection
    4. PCA
    5. Neighbor graph + UMAP
    6. Leiden clustering

    Parameters
    ----------
    adata
        AnnData object with raw counts
    min_genes
        Minimum number of genes per cell
    min_cells
        Minimum number of cells per gene
    max_pct_mito
        Maximum percentage of mitochondrial reads
    n_top_genes
        Number of highly variable genes to select
    target_sum
        Target sum for normalization
    n_pcs
        Number of principal components
    n_neighbors
        Number of neighbors for KNN graph
    batch_key
        Column in obs for batch correction (uses Harmony if provided)
    copy
        Whether to operate on a copy

    Returns
    -------
    Preprocessed AnnData
    """
    if copy:
        adata = adata.copy()

    logger.info(f"Starting preprocessing: {adata.n_obs} cells, {adata.n_vars} genes")

    # Store raw counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter cells
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
    logger.info(f"Filtered cells: {n_before} -> {adata.n_obs}")

    # Filter genes
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"Filtered genes: {n_genes_before} -> {adata.n_vars}")

    # Normalize
    adata = normalize(adata, target_sum=target_sum)

    # HVG selection
    adata = select_hvgs(adata, n_top_genes=n_top_genes, batch_key=batch_key)

    # PCA
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    # Batch correction with Harmony (if batch_key provided)
    if batch_key and batch_key in adata.obs.columns:
        logger.info(f"Applying Harmony batch correction on '{batch_key}'")
        sc.external.pp.harmony_integrate(adata, batch_key, basis="X_pca", adjusted_basis="X_pca_harmony")
        use_rep = "X_pca_harmony"
    else:
        use_rep = "X_pca"

    # Neighbors + UMAP + Leiden
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)

    logger.info(
        f"Preprocessing complete: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs['leiden'].nunique()} clusters"
    )

    return adata


def normalize(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    copy: bool = False,
) -> ad.AnnData:
    """
    Normalize and log-transform counts.

    Parameters
    ----------
    adata
        AnnData with raw counts
    target_sum
        Target library size for normalization
    copy
        Whether to operate on a copy

    Returns
    -------
    Normalized AnnData
    """
    if copy:
        adata = adata.copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Store normalized data
    adata.raw = adata

    return adata


def select_hvgs(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    batch_key: Optional[str] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Select highly variable genes.

    Parameters
    ----------
    adata
        Normalized AnnData
    n_top_genes
        Number of top highly variable genes
    batch_key
        Batch key for batch-aware HVG selection
    copy
        Whether to operate on a copy

    Returns
    -------
    AnnData filtered to HVGs
    """
    if copy:
        adata = adata.copy()

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        flavor="seurat_v3" if "counts" in adata.layers else "seurat",
        layer="counts" if "counts" in adata.layers else None,
    )

    n_hvgs = adata.var["highly_variable"].sum()
    logger.info(f"Selected {n_hvgs} highly variable genes")

    adata = adata[:, adata.var["highly_variable"]].copy()

    return adata
