"""
This package provides classification models for the AutoML framework.

Available models:
- Logistic
- KNearestNeighbors
- SupportVectorMachine
"""

from .logistic_regression_model import Logistic
from .k_nearest_neighbors_model import KNearestNeighbors
from .svm_model import SupportVectorMachine

__all__ = [
    "Logistic",
    "KNearestNeighbors",
    "SupportVectorMachine",
]
