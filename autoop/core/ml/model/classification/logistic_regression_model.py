from sklearn.linear_model import LogisticRegression
import numpy as np
from autoop.core.ml.model.model import Model


class Logistic(Model):
    """
    Wrapper for the Logistic Regression classifier.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the Logistic Regression model with
        optional hyperparameters.

        Args:
            kwargs: Additional keyword arguments for Logistic Regression model.
        """
        super().__init__()
        self._model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the logistic regression model using the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self._model.fit(X, y)
        self.parameters = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target classes using the logistic regression model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self._model.predict(X)
