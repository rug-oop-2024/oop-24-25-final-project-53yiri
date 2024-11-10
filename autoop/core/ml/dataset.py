from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
from typing import Optional


class Dataset(Artifact):
    """
    A class to represent an ML dataset, inheriting from Artifact.

    Attributes:
        name (str): The name of the dataset.
        asset_path (str): The path where the dataset is stored.
        version (str): The version of the dataset.
        data (bytes): The encoded dataset data.
    """

    def __init__(
        self, name: str, asset_path: str,
        version: str = "1.0.0", data: Optional[bytes] = None
    ):
        """
        Initialize the Dataset with a name and additional attributes.

        Args:
            name (str): The name of the dataset.
            asset_path (str): Path where the asset is stored.
            version (str): Version of the dataset artifact.
            data (Optional[bytes]): Encoded dataset data.
        """
        super().__init__(
            name=name,
            asset_path=asset_path,
            version=version,
            data=data,
            metadata={},
            type="dataset"
        )

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The data to store in the Dataset.
            name (str): The name of the dataset.
            asset_path (str): Path where the asset is stored.
            version (str): Version of the dataset artifact.

        Returns:
            Dataset: The created Dataset instance.
        """
        dataset_data = data.to_csv(index=False).encode()
        return Dataset(
            name=name,
            asset_path=asset_path,
            version=version,
            data=dataset_data
        )

    def read(self) -> pd.DataFrame:
        """Reads the dataset's data."""
        if isinstance(self.data, bytes):
            csv_data = self.data.decode()
            return pd.read_csv(io.StringIO(csv_data))
        else:
            raise TypeError("Dataset data must be in bytes format.")
