from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """A class to represent an ML dataset, inheriting from Artifact."""

    def __init__(self, *args, **kwargs):
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str,
        version: str = "1.0.0"
                       ) -> "Dataset":
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The data to store in the Dataset.
            name (str): The name of the dataset.
            asset_path (str): Path where the asset is stored.
            version (str): Version of the artifact.

        Returns:
            Dataset: The created Dataset instance.
        """
        return Dataset(
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            metadata={"name": name}  # Store the name in metadata
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset's data, decoding
        from bytes to a DataFrame if necessary.

        Returns:
            pd.DataFrame: DataFrame containing the dataset.
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data  # Return directly if already a DataFrame
        elif isinstance(self.data, bytes):
            # Decode and read from bytes
            csv_data = self.data.decode()
            return pd.read_csv(io.StringIO(csv_data))
        else:
            raise TypeError("Dataset data is neither a DataFrame"
                            "nor in bytes format.")

    def save(self, data: pd.DataFrame) -> None:
        """Saves a DataFrame to the Dataset by encoding it as bytes."""
        self.data = data.to_csv(index=False).encode()
