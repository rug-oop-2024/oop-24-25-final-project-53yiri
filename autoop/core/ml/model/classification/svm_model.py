from sklearn.svm import SVC
import numpy as np
from autoop.core.ml.model.model import Model


class SupportVectorMachine(Model):
    """
    Wrapper for the Support Vector Machine (SVM) classifier with RBF kernel.
    """

    def __init__(self, C: float = 1.0, gamma: str = 'scale', **kwargs) -> None:
        """
        Initializes the SVM model with an RBF kernel.

        Args:
            C (float): Regularization parameter.
            gamma (str or float): Kernel coefficient.
            kwargs: Additional keyword arguments for the SVM model.
        """
        super().__init__()
        self._model = SVC(kernel='rbf', C=C, gamma=gamma, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVM model on the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self._model.fit(X, y)
        self.parameters = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target classes using the SVM model.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self._model.predict(X)
