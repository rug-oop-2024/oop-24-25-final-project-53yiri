from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    Pipeline class for managing the machine learning workflow.

    Attributes:
        _dataset (Dataset): The dataset to be used in the pipeline.
        _model (Model): The machine learning model to be trained and evaluated.
        _input_features (List[Feature]): A list of features to be used as
                                        input for the model.
        _target_feature (Feature): The feature to be predicted by the model.
        _metrics (List[Metric]): A list of metrics to evaluate the model.
        _artifacts (dict): A dictionary to store artifacts generated
                            during the pipeline execution.
        _split (float): The ratio of dataset used for training. Default is 0.8.
        _output_vector (np.array): The target feature data.
        _input_vectors (List[np.array]): The input features data.
        _train_X (List[np.array]): The training input data.
        _test_X (List[np.array]): The testing input data.
        _train_y (np.array): The training target data.
        _test_y (np.array): The testing target data.
        _metrics_results (List[Tuple[Metric, float]]): The results of
                                                        the metrics evaluation.
        _predictions (np.array): The predictions made by the model.

    Methods:
        __init__(self, metrics: List[Metric], dataset: Dataset, model: Model,
            input_features: List[Feature], target_feature: Feature, split=0.8):
        __str__(self) -> str:
            Returns a string representation of the pipeline.
        model(self) -> Model:
        artifacts(self) -> List[Artifact]:
            Returns the artifacts generated during the pipeline execution.
        _register_artifact(self, name: str, artifact):
            Registers an artifact with the given name.
        _preprocess_features(self):
            Preprocesses the input and target features.
        _split_data(self):
            Splits the data into training and testing sets.
        _compact_vectors(self, vectors: List[np.array]) -> np.array:
            Compacts a list of vectors into a single numpy array.
        _train(self):
            Trains the machine learning model.
        _evaluate(self):
            Evaluates the machine learning model.
        execute(self) -> dict:
            Executes the pipeline and returns the results.
        """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the pipeline with the given parameters.

        Args:
            metrics (List[Metric]): A list of metrics to evaluate the model.
            dataset (Dataset): The dataset to be used in the pipeline.
            model (Model): The machine learning model to be trained
                            and evaluated.
            input_features (List[Feature]): A list of features to be used as
                                            input for the model.
            target_feature (Feature): The feature to be predicted by the model.
            split (float): The ratio of dataset used for training. Default 0.8.

        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the machine learning model.

        Returns:
            object: The machine learning model instance.
        """
        """"""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the
        pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: object) -> None:
        """Registers an artifact with the given name."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Compacts a list of vectors into a single numpy array."""
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """Execute the pipeline."""
        # Step 1: Read the original dataset
        data = self._dataset.read()

        # Step 2: Extract input and target feature names
        input_feature_names = [
            feature.name for feature in self._input_features
        ]
        target_feature_name = self._target_feature.name

        # Step 3: Select relevant columns (assuming no missing values)
        selected_columns = input_feature_names + [target_feature_name]
        df = data[selected_columns]

        # Step 4: Create a temporary dataset for further processing
        from autoop.core.ml.dataset import Dataset
        temp_dataset = Dataset.from_dataframe(
            data=df,
            name=self._dataset.name,
            asset_path=self._dataset.asset_path,
            version=self._dataset.version
        )
        self._dataset = temp_dataset

        # Step 5: Preprocess features
        self._preprocess_features()

        # Step 6: Process the target variable based on feature type
        target_series = df[target_feature_name]
        if self._target_feature.feature_type == "categorical":
            # Encode categorical target variable for classification tasks
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            self._output_vector = label_encoder.fit_transform(target_series)
            self._register_artifact(
                'label_encoder', {'type': 'LabelEncoder',
                                  'encoder': label_encoder}
            )
        else:
            # Use the raw values for regression tasks
            self._output_vector = target_series.to_numpy()

        # Step 7: Split data into training and test sets
        self._split_data()

        # Step 8: Train the model
        self._train()

        # Step 9: Evaluate the model on the training set
        train_X = self._compact_vectors(self._train_X)
        train_Y = self._train_y
        train_predictions = self._model.predict(train_X)
        train_metrics_results = {}
        for metric in self._metrics:
            metric_name = metric.__class__.__name__
            train_metric_score = metric.evaluate(train_predictions, train_Y)
            train_metrics_results[metric_name] = train_metric_score

        # Step 10: Evaluate the model on the test set
        test_X = self._compact_vectors(self._test_X)
        test_Y = self._test_y
        test_predictions = self._model.predict(test_X)
        test_metrics_results = {}
        for metric in self._metrics:
            metric_name = metric.__class__.__name__
            test_metric_score = metric.evaluate(test_predictions, test_Y)
            test_metrics_results[metric_name] = test_metric_score

        # Return training and test metrics along with predictions
        return {
            "metrics": {
                'training': train_metrics_results,
                'evaluation': test_metrics_results,
            },
            "predictions": {
                "train": train_predictions,
                "test": test_predictions,
            },
            "model": self._model
        }
