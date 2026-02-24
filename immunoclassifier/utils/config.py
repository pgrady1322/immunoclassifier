"""
YAML configuration handling for training and benchmarking runs.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to YAML config file

    Returns
    -------
    Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config
