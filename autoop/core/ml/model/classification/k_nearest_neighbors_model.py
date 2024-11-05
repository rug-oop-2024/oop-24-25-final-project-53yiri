from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from autoop.core.ml.model.model import Model


class KNearestNeighbors(Model):
    """
    Wrapper for the K-Nearest Neighbors (KNN) classifier.
    """

    def __init__(self, n_neighbors: int = 5, **kwargs) -> None:
        """
        Initializes the K-Nearest Neighbors classifier with
        optional hyperparameters.

        Args:
            n_neighbors (int): Number of neighbors to use.
            kwargs: Additional keyword arguments for KNN model.
        """
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the K-Nearest Neighbors model on the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target classes using the K-Nearest Neighbors model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(X)
