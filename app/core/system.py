from autoop.core.storage import LocalStorage, Storage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from typing import List, Optional


class ArtifactRegistry:
    """
    Manages the registration, retrieval, and deletion of artifacts.

    Attributes:
        _database (Database): The database for
                            storing artifact metadata.
        _storage (Storage): The storage system for
                            saving and loading artifact data.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initializes the ArtifactRegistry with a
        specified database and storage system.

        Args:
            database (Database): The database instance
                                to store artifact metadata.
            storage (Storage): The storage instance to store artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers a new artifact by saving its data
        in storage and metadata in the database.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        # Save the artifact data in the storage
        self._storage.save(artifact.data, artifact.asset_path)

        # Save the artifact metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: Optional[str] = None) -> List[Artifact]:
        """
        Retrieves a list of artifacts, optionally filtered by type.

        Args:
            type (Optional[str]): The type of artifacts to filter by.
                                Defaults to None.

        Returns:
            List[Artifact]: A list of artifacts matching the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for artifact_id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            Artifact: The artifact with the specified ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID from both the storage and the database.

        Args:
            artifact_id (str): The unique identifier of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    A singleton class representing the AutoML system that manages storage,
    database, and artifact registry.

    Attributes:
        _instance (Optional[AutoMLSystem]): The singleton instance
                                            of AutoMLSystem.
        _storage (Storage): The storage system for artifacts.
        _database (Database): The database for artifact metadata.
        _registry (ArtifactRegistry): The registry managing
                                            artifact operations.
    """
    _instance: Optional["AutoMLSystem"] = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoMLSystem with specified storage and database.

        Args:
            storage (LocalStorage): The storage instance for artifacts.
            database (Database): The database instance for artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieves the singleton instance of AutoMLSystem,
        creating it if it doesn't exist.

        Returns:
            AutoMLSystem: The singleton instance of the AutoML system.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Provides access to the artifact registry.

        Returns:
            ArtifactRegistry: The registry managing artifact operations.
        """
        return self._registry
