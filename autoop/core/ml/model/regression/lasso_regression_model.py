from sklearn.linear_model import Lasso
import numpy as np
from autoop.core.ml.model.model import Model


class LassoRegression(Model):
    """
    Wrapper for the Lasso Regression model.
    """

    def __init__(self, alpha: float = 10.0, **kwargs) -> None:
        """
        Initializes the Lasso Regression model with L1 regularization.

        Args:
            alpha (float): Regularization strength.
            kwargs: Additional keyword arguments for Lasso Regression model.
        """
        super().__init__()
        self._model = Lasso(alpha=alpha, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the lasso regression model using the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self._model.fit(X, y.ravel())
        self.parameters = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the lasso regression model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(X)
