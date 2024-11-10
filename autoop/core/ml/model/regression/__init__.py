from .multiple_linear_regression_model import MultipleLinearRegression
from .lasso_regression_model import LassoRegression
from .random_forest_regressor_model import RandomForest
"""
This package provides regression models for the AutoML framework.

Available models:
- MultipleLinearRegression
- LassoRegression
- RandomForest
"""

__all__ = [
    "MultipleLinearRegression",
    "LassoRegression",
    "RandomForest",
]
