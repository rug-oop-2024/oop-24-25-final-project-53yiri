from sklearn.ensemble import RandomForestRegressor
import numpy as np
from autoop.core.ml.model.model import Model


class RandomForest(Model):
    """
    Wrapper for the Random Forest Regressor model.
    """

    def __init__(self, n_estimators: int = 1000, **kwargs) -> None:
        """
        Initializes the Random Forest Regressor model.

        Args:
            n_estimators (int): Number of trees in the forest.
            kwargs: Additional keyword arguments for
            Random Forest Regressor model.
        """
        super().__init__()
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, **kwargs
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the random forest regressor on the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self._model.fit(X, y.ravel())
        self.parameters = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the random forest regressor.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._model.predict(X)
