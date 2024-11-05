from pydantic import BaseModel, Field, root_validator
from typing import Literal


class Feature(BaseModel):
    """
    Represents a feature in the dataset, characterized by its name and type.

    Attributes:
        name (str): The name of the feature.
        feature_type (Literal['categorical', 'numerical']): The type of the
        feature, either categorical or numerical.
        unique_values (int): The number of unique values for the feature.
    """

    name: str = Field(..., description="Name of the feature.")
    feature_type: Literal['categorical', 'numerical'] = Field(
        None, description="Type of the feature."
    )
    unique_values: int = Field(
        0, ge=0, description="Number of unique values in the feature."
    )

    @root_validator(pre=True)
    def handle_test_compatibility(cls, values):
        """
        Validator to support compatibility with test cases that
        use 'type' instead of 'feature_type' and may omit 'unique_values'.
        """
        if 'type' in values and 'feature_type' not in values:
            values['feature_type'] = values.pop('type')
        if 'unique_values' not in values:
            values['unique_values'] = 0
        return values

    @property
    def type(self) -> str:
        """
        Alias for `feature_type`.

        Returns:
            str: The type of the feature (categorical or numerical).
        """
        return self.feature_type

    def __str__(self) -> str:
        """
        Provides a string representation of the Feature instance,
        summarizing its name and type.

        Returns:
            str: A string describing the feature's name and type.
        """
        return (
            f"Feature(name={self.name}, type={self.feature_type}, "
            f"unique_values={self.unique_values})"
        )
