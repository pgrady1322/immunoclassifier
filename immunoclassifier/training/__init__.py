#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmunoClassifier v0.1.0

Training subpackage — trainer and hyperparameter optimization.

Author: Patrick Grady
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: MIT License - See LICENSE
"""

from immunoclassifier.training.trainer import Trainer
from immunoclassifier.training.hyperopt import run_hyperopt, train_with_best_params

__all__ = ["Trainer", "run_hyperopt", "train_with_best_params"]

# ImmunoClassifier v0.1.0
# Any usage is subject to this software's license.
