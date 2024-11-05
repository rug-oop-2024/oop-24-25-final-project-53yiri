import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict
from copy import deepcopy
import json
import pickle
from pydantic import BaseModel, PrivateAttr


class Model(ABC, BaseModel):
    """
    Abstract base class for all models, defining the interface for fitting,
    predicting, and evaluating, with support for parameter storage
    and management.

    Attributes:
        _parameters (Dict[str, Any]): A dictionary to store model parameters
        and hyperparameters.
    """

    _parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.

        Args:
            X (np.ndarray): The observed values (features).
            y (np.ndarray): The target values (labels).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted model.

        Args:
            X (np.ndarray): The observed values (features).

        Returns:
            np.ndarray: Predicted values.
        """
        pass

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluates the model on given data and returns metric results.

        Args:
            X (np.ndarray): Features for evaluation.
            y (np.ndarray): Ground truth labels or values.
            metrics (Dict[str, Any]): Dictionary of metric instances,
            where each metric is callable.

        Returns:
            Dict[str, float]: Calculated metric scores.
        """
        predictions = self.predict(X)
        return {name: metric(y, predictions) for name,
                metric in metrics.items()}

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, value: Dict[str, Any]) -> None:
        """Set model parameters with validation."""
        if not isinstance(value, dict):
            raise ValueError("Parameters must be a dictionary.")
        self._parameters = value

    def save(self, file_path: str, method: str = 'pickle') -> None:
        """
        Saves the model parameters to a file.

        Args:
            file_path (str): Path to save the model.
            method (str): Serialization method ('pickle' or 'json').
        """
        if method == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(self._parameters, f)
        elif method == 'json':
            with open(file_path, 'w') as f:
                json.dump(self._parameters, f)
        else:
            raise ValueError(
                "Unsupported save method. Use 'pickle' or 'json'."
                )

    def load(self, file_path: str, method: str = 'pickle') -> None:
        """
        Loads model parameters from a file and restores the model's state.

        Args:
            file_path (str): Path to load the model from.
            method (str): Deserialization method ('pickle' or 'json').
        """
        if method == 'pickle':
            with open(file_path, 'rb') as f:
                self._parameters = pickle.load(f)
        elif method == 'json':
            with open(file_path, 'r') as f:
                self._parameters = json.load(f)
        else:
            raise ValueError(
                "Unsupported load method. Use 'pickle' or 'json'."
                )
