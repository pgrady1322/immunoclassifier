#!/usr/bin/env python3
"""
ImmunoClassifier — ML-powered immune cell type classification from scRNA-seq data.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="immunoclassifier",
    version="0.1.0",
    description="ML-powered immune cell type classification from scRNA-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Patrick Grady",
    url="https://github.com/pgrady3/immunoclassifier",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0",
        "scanpy>=1.9",
        "anndata>=0.10",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "torch>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scvi-tools>=1.0",
        "pyyaml",
        "tqdm",
        "rich",
    ],
    extras_require={
        "gnn": ["torch-geometric>=2.4"],
        "foundation": ["gdown"],
        "benchmark": ["celltypist"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "isort", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "immunoclassifier=immunoclassifier.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
