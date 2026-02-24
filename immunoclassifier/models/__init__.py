#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmunoClassifier v0.1.0

Models subpackage — all classifier implementations.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from immunoclassifier.models.base import BaseClassifier
from immunoclassifier.models.logistic import LogisticClassifier
from immunoclassifier.models.xgboost_model import XGBoostClassifier
from immunoclassifier.models.scvi_classifier import ScVIClassifier
from immunoclassifier.models.gnn_classifier import GNNClassifier

__all__ = [
    "BaseClassifier",
    "LogisticClassifier",
    "XGBoostClassifier",
    "ScVIClassifier",
    "GNNClassifier",
]

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
