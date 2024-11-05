from abc import ABC, abstractmethod
import numpy as np

# List of available metric names
METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall"
]


def get_metric(name: str) -> "Metric":
    """
    Factory function to retrieve a metric instance by its name.

    Args:
        name (str): The name of the metric.

    Returns:
        Metric: An instance of the specified metric class.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    metric_map = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "r2_score": R2Score(),
        "accuracy": Accuracy(),
        "precision": Precision(),
        "recall": Recall()
    }
    if name in metric_map:
        return metric_map[name]
    else:
        raise ValueError(
            f"Metric '{name}' is not implemented. Available metrics: {METRICS}"
            )


class Metric(ABC):
    """
    Abstract base class for all metrics.

    Each metric calculates a score based on ground truth and predictions,
    returning a real number that reflects the performance of the model.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the metric given ground truth and predictions.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The calculated metric value.
        """
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluates the metric by calling the __call__ method.

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The calculated metric value.
        """
        return self(y_true, y_pred)


# Regression Metrics

class MeanSquaredError(Metric):
    """Calculates the Mean Squared Error (MSE) for regression tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class MeanAbsoluteError(Metric):
    """Calculates the Mean Absolute Error (MAE) for regression tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class R2Score(Metric):
    """Calculates the R-squared (R2) score for regression tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


# Classification Metrics

class Accuracy(Metric):
    """Calculates accuracy for classification tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class Precision(Metric):
    """Calculates precision for binary classification tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        return true_positive / (true_positive + false_positive + 1e-10)


class Recall(Metric):
    """Calculates recall for binary classification tasks."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_negative = np.sum((y_pred == 0) & (y_true == 1))
        return true_positive / (true_positive + false_negative + 1e-10)
