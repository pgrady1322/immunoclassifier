#!/usr/bin/env python3
"""
ImmunoClassifier v0.1.0

YAML configuration loader and validation.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict[str, Any]:
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

    with open(path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
