from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found.

    Attributes:
        path (str): The path that was not found.
    """
    def __init__(self, path: str):
        """
        Custom exception raised when a specified path is not found.

        Args:
            path (str): Path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for storage systems. Defines methods for saving,
    loading, deleting, and listing data at specific paths.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a specified path.

        Args:
            data (bytes): Data to be saved.
            path (str): Path where data will be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a specified path.

        Args:
            path (str): Path from which data will be loaded.

        Returns:
            bytes: The data loaded from the specified path.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a specified path.

        Args:
            path (str): Path of the data to be deleted.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all files under a specified path.

        Args:
            path (str): Path to list files from.

        Returns:
            List[str]: List of file paths.
        """
        pass


class LocalStorage(Storage):
    """
    Concrete implementation of the Storage abstract base class, using the local
    file system as the storage medium.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a specified base directory. Creates the
        directory if it does not exist.

        Args:
            base_path (str): The base directory for storage.
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a specified location within the base path.

        Args:
            data (bytes): The data to be saved.
            key (str): The relative path from base_path where
                    the data will be saved.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a specified location within the base path.

        Args:
            key (str): The relative path from base_path
                        where the data is stored.

        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a specified location within the base path.

        Args:
            key (str): The relative path from
                        base_path where the data is stored.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all files under a specified prefix within the base path.

        Args:
            prefix (str): The relative path prefix to list files from.

        Returns:
            List[str]: A list of file paths under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return [key for key in keys if os.path.isfile(key)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a specified path exists. Raise NotFoundError if it does not.

        Args:
            path (str): The path to check for existence.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join a relative path with the base path to get an absolute path.

        Args:
            path (str): The relative path.

        Returns:
            str: The absolute path by joining base_path and the specified path.
        """
        return os.path.join(self._base_path, path)
