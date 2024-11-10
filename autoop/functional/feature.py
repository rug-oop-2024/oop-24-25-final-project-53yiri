import pandas as pd
from io import BytesIO
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detects and categorizes features in the provided Dataset object as either
    'categorical' or 'numerical', based on the feature type.

    This function checks if `dataset.data` is in bytes format and converts it
    to a DataFrame if necessary. Features are classified as 'numerical' based
    on data type only (integer or float); otherwise, they are 'categorical'.

    Args:
        dataset (Dataset): The dataset containing features to be analyzed.

    Returns:
        List[Feature]: A list of Feature objects,
        each with name, type, and unique values.
    """
    # Convert bytes to DataFrame if necessary
    if isinstance(dataset.data, bytes):
        data = pd.read_csv(BytesIO(dataset.data))
    else:
        data = dataset.data

    features = []

    # Detect and classify features based on data type only
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            feature_type = 'numerical'
        else:
            feature_type = 'categorical'

        unique_values_count = data[column].nunique()
        features.append(
            Feature(name=column, feature_type=feature_type,
                    unique_values=unique_values_count
                    )
        )

    return features
