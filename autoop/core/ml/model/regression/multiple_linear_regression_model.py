from sklearn.linear_model import LinearRegression
from pydantic import PrivateAttr
import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    Wrapper for the Multiple Linear Regression model.
    """

    _model: LinearRegression = PrivateAttr()

    def __init__(self, **kwargs) -> None:
        """
        Initializes the Multiple Linear Regression model with
        optional hyperparameters.

        Args:
            kwargs: Additional keyword arguments for Linear Regression model.
        """
        super().__init__()
        self._model = LinearRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the multiple linear regression model using the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self._model.fit(X, y.ravel())
        self.parameters = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the multiple linear regression model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(X)

