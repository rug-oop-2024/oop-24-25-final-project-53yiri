from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import base64


class Artifact(BaseModel):
    """
    Represents an artifact in the AutoML system,
    encapsulating details about a dataset,
    model, or other assets used within the pipeline.
    """

    name: str = Field(..., description="Name of the artifact.")
    asset_path: str = Field(
        ..., description="Path to the asset in the storage system."
    )
    version: str = Field(..., description="Version of the artifact.")
    data: Any = Field(
        ...,
        description="Binary or object data representing artifact's content."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with artifact."
    )
    type: Optional[str] = Field(
        None, description="Type of the artifact."
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags that categorize artifact."
    )

    @property
    def id(self) -> str:
        """
        Generates a unique identifier for the artifact by
        encoding the asset path and combining it with the version.

        Returns:
            str: The unique identifier for the artifact.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves the metadata dictionary associated with the artifact.

        Returns:
            Dict[str, Any]: A dictionary containing metadata for the artifact.
        """
        return self.metadata

    def update_metadata(self, key: str, value: Any) -> None:
        """
        Updates or adds a key-value pair in the artifact's metadata.

        Args:
            key (str): The metadata key to update.
            value (Any): The new value for the specified metadata key.
        """
        self.metadata[key] = value
