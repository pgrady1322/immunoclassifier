"""
Dataset loaders for immune cell classification.

Provides standardized access to public single-cell immune datasets
with consistent cell type annotations.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "immunoclassifier" / "data"

# Available datasets registry
DATASETS = {
    "pbmc_10k": {
        "description": "10x Genomics PBMC 10k dataset",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/6.1.0/10k_PBMC_3p_nextgem_Chromium_Controller/10k_PBMC_3p_nextgem_Chromium_Controller_filtered_feature_bc_matrix.h5",
        "n_cells": "~10,000",
        "reference": "10x Genomics (2022)",
    },
    "tabula_sapiens_immune": {
        "description": "Tabula Sapiens immune compartment (blood, spleen, lymph node, bone marrow)",
        "url": "https://figshare.com/ndownloader/files/34701991",
        "n_cells": "~100,000",
        "reference": "Tabula Sapiens Consortium (2022) Science",
    },
    "hao_cite_seq": {
        "description": "Hao et al. CITE-seq PBMC (161k cells, protein-validated annotations)",
        "url": None,  # Requires GEO download
        "geo_accession": "GSE164378",
        "n_cells": "~161,000",
        "reference": "Hao et al. (2021) Cell",
    },
}

# Standardized immune cell type hierarchy
IMMUNE_CELL_HIERARCHY = {
    "CD4+ T cells": [
        "CD4+ Naive T",
        "CD4+ Central Memory T",
        "CD4+ Effector Memory T",
        "Th1",
        "Th2",
        "Th17",
        "Treg",
    ],
    "CD8+ T cells": [
        "CD8+ Naive T",
        "CD8+ Effector T",
        "CD8+ Memory T",
        "CD8+ Exhausted T",
    ],
    "B cells": [
        "Naive B",
        "Memory B",
        "Plasma cell",
        "Plasmablast",
    ],
    "NK cells": [
        "NK CD56bright",
        "NK CD56dim",
    ],
    "Monocytes": [
        "Classical Monocyte",
        "Non-classical Monocyte",
        "Intermediate Monocyte",
    ],
    "Dendritic cells": [
        "cDC1",
        "cDC2",
        "pDC",
    ],
    "Other": [
        "Macrophage",
        "Platelet",
        "HSPC",
    ],
}

# Key marker genes for immune cell types
IMMUNE_MARKERS = {
    "CD4+ T cells": ["CD3E", "CD3D", "CD4", "IL7R"],
    "CD8+ T cells": ["CD3E", "CD3D", "CD8A", "CD8B"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "Th17": ["RORC", "IL17A", "CCR6", "IL23R"],
    "CD8+ Exhausted T": ["PDCD1", "LAG3", "HAVCR2", "TOX"],
    "B cells": ["CD19", "MS4A1", "CD79A", "CD79B"],
    "Plasma cell": ["SDC1", "MZB1", "XBP1", "JCHAIN"],
    "NK cells": ["NCAM1", "NKG7", "KLRD1", "GNLY"],
    "NK CD56bright": ["NCAM1", "SELL", "GZMK", "XCL1"],
    "NK CD56dim": ["FCGR3A", "PRF1", "GZMB", "FGFBP2"],
    "Classical Monocyte": ["CD14", "LYZ", "S100A9", "VCAN"],
    "Non-classical Monocyte": ["FCGR3A", "MS4A7", "CDKN1C", "LILRB2"],
    "cDC1": ["CLEC9A", "XCR1", "BATF3", "IRF8"],
    "cDC2": ["CD1C", "FCER1A", "CLEC10A", "HLA-DQA1"],
    "pDC": ["LILRA4", "IRF7", "CLEC4C", "IL3RA"],
    "HSPC": ["CD34", "KIT", "CD38"],
}


def list_available_datasets() -> dict:
    """List all available datasets with descriptions."""
    return {k: v["description"] for k, v in DATASETS.items()}


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def load_pbmc_10k(cache_dir: Optional[str] = None, force_download: bool = False) -> ad.AnnData:
    """
    Load the 10x Genomics PBMC 10k dataset.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded data. Defaults to ~/.cache/immunoclassifier/data
    force_download
        If True, re-download even if cached file exists

    Returns
    -------
    AnnData object with raw counts
    """
    cache = Path(cache_dir) if cache_dir else _ensure_cache_dir()
    filepath = cache / "pbmc_10k.h5ad"

    if filepath.exists() and not force_download:
        logger.info(f"Loading cached PBMC 10k data from {filepath}")
        return sc.read_h5ad(filepath)

    logger.info("Downloading PBMC 10k dataset from 10x Genomics...")

    # Download the filtered feature barcode matrix
    h5_path = cache / "pbmc_10k_raw.h5"
    if not h5_path.exists() or force_download:
        import urllib.request

        url = DATASETS["pbmc_10k"]["url"]
        urllib.request.urlretrieve(url, h5_path)
        logger.info(f"Downloaded to {h5_path}")

    # Read and convert
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    # Store raw counts
    adata.layers["counts"] = adata.X.copy()

    # Save processed
    adata.write_h5ad(filepath)
    logger.info(f"Saved to {filepath}")

    return adata


def load_tabula_sapiens_immune(
    cache_dir: Optional[str] = None, force_download: bool = False
) -> ad.AnnData:
    """
    Load the Tabula Sapiens immune compartment.

    Includes cells from blood, spleen, lymph node, and bone marrow
    with expert-curated cell type annotations.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded data
    force_download
        If True, re-download even if cached file exists

    Returns
    -------
    AnnData object with raw counts and cell type annotations in obs['cell_type']
    """
    cache = Path(cache_dir) if cache_dir else _ensure_cache_dir()
    filepath = cache / "tabula_sapiens_immune.h5ad"

    if filepath.exists() and not force_download:
        logger.info(f"Loading cached Tabula Sapiens immune data from {filepath}")
        return sc.read_h5ad(filepath)

    logger.info("Downloading Tabula Sapiens immune compartment...")
    logger.info(
        "Note: This is a large download (~2GB). "
        "You may also download manually from https://tabula-sapiens-portal.ds.czbiohub.org/"
    )

    # Download full Tabula Sapiens and filter to immune compartment
    url = DATASETS["tabula_sapiens_immune"]["url"]
    ts_path = cache / "tabula_sapiens_full.h5ad"

    if not ts_path.exists() or force_download:
        import urllib.request

        urllib.request.urlretrieve(url, ts_path)

    adata = sc.read_h5ad(ts_path)

    # Filter to immune-relevant organs
    immune_organs = ["Blood", "Spleen", "Lymph_Node", "Bone_Marrow"]
    if "organ_tissue" in adata.obs.columns:
        mask = adata.obs["organ_tissue"].isin(immune_organs)
        adata = adata[mask].copy()
    elif "tissue" in adata.obs.columns:
        mask = adata.obs["tissue"].isin(immune_organs)
        adata = adata[mask].copy()

    logger.info(f"Filtered to {adata.n_obs} immune cells from {immune_organs}")

    # Standardize cell type column
    for col in ["cell_ontology_class", "cell_type", "free_annotation"]:
        if col in adata.obs.columns:
            adata.obs["cell_type"] = adata.obs[col].copy()
            break

    adata.write_h5ad(filepath)
    logger.info(f"Saved to {filepath}")

    return adata


def load_hao_cite_seq(
    cache_dir: Optional[str] = None, force_download: bool = False
) -> ad.AnnData:
    """
    Load Hao et al. 2021 CITE-seq PBMC dataset (GSE164378).

    This dataset has protein-validated cell type annotations, making it
    ideal for benchmarking RNA-only classification methods.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded data
    force_download
        If True, re-download even if cached file exists

    Returns
    -------
    AnnData object with raw counts and cell type annotations
    """
    cache = Path(cache_dir) if cache_dir else _ensure_cache_dir()
    filepath = cache / "hao_cite_seq.h5ad"

    if filepath.exists() and not force_download:
        logger.info(f"Loading cached Hao CITE-seq data from {filepath}")
        return sc.read_h5ad(filepath)

    logger.info(
        "Hao et al. CITE-seq PBMC dataset requires manual download from GEO.\n"
        f"  GEO accession: {DATASETS['hao_cite_seq']['geo_accession']}\n"
        "  Steps:\n"
        "  1. Download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378\n"
        "  2. Or use the SeuratData R package: `library(SeuratData); InstallData('pbmcMultiome')`\n"
        "  3. Export as h5ad and place at: {filepath}\n"
    )
    raise FileNotFoundError(
        f"Hao CITE-seq data not found at {filepath}. See instructions above for manual download."
    )
