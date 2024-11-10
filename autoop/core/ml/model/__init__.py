"""
This package provides machine learning models.
"""

from .classification import KNearestNeighbors, Logistic, SupportVectorMachine
from .regression import LassoRegression, MultipleLinearRegression, RandomForest

__all__ = [
    "KNearestNeighbors",
    "Logistic",
    "SupportVectorMachine",
    "LassoRegression",
    "MultipleLinearRegression",
    "RandomForest",
]
