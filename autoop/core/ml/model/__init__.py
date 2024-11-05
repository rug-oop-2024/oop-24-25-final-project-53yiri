# Import the base model and specific models
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.logistic_regression_model import (
    Logistic
)
from autoop.core.ml.model.classification.k_nearest_neighbors_model import (
    KNearestNeighbors
)
from autoop.core.ml.model.classification.svm_model import (
    SupportVectorMachine
)
from autoop.core.ml.model.regression.multiple_linear_regression_model import (
    MultipleLinearRegression
)
from autoop.core.ml.model.regression.lasso_regression_model import (
    LassoRegression
)
from autoop.core.ml.model.regression.random_forest_regressor_model import (
    RandomForest
)

# Lists of available models by name
REGRESSION_MODELS = [
    "MultipleLinearRegressionModel",
    "LassoRegressionModel",
    "RandomForestRegressorModel"
]

CLASSIFICATION_MODELS = [
    "LogisticRegressionModel",
    "KNearestNeighborsModel",
    "SupportVectorMachineModel"
]


def get_model(model_name: str, **kwargs) -> Model:
    """
    Factory function to get a model instance by name.

    Args:
        model_name (str): Name of the model to instantiate.
        kwargs: Additional parameters to pass to the model constructor.

    Returns:
        Model: An instance of the specified model class.

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_map = {
        "MultipleLinearRegressionModel": MultipleLinearRegression,
        "LassoRegressionModel": LassoRegression,
        "RandomForestRegressorModel": RandomForest,
        "LogisticRegressionModel": Logistic,
        "KNearestNeighborsModel": KNearestNeighbors,
        "SupportVectorMachineModel": SupportVectorMachine,
    }

    if model_name in model_map:
        return model_map[model_name](**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized."
                         "Available models are: "
                         f"{REGRESSION_MODELS + CLASSIFICATION_MODELS}")
