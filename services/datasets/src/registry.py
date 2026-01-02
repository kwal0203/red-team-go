"""Dataset registry for managing available datasets.

Provides a central registry for discovering and loading datasets.
"""

import logging

from services.datasets.src.base import (
    BaseDatasetLoader,
    DatasetCategory,
    DatasetInfo,
    DatasetNotFoundError,
)

logger = logging.getLogger(__name__)

# Global registry of dataset loaders
_DATASET_REGISTRY: dict[str, type[BaseDatasetLoader]] = {}


def register_dataset(loader_class: type[BaseDatasetLoader]) -> type[BaseDatasetLoader]:
    """Decorator to register a dataset loader.

    Args:
        loader_class: The dataset loader class to register.

    Returns:
        The same class (allows use as decorator).
    """
    # Create instance to get info
    instance = loader_class()
    name = instance.info.name

    if name in _DATASET_REGISTRY:
        logger.warning(f"Dataset '{name}' already registered, overwriting")

    _DATASET_REGISTRY[name] = loader_class
    logger.debug(f"Registered dataset: {name}")

    return loader_class


def get_dataset(name: str) -> BaseDatasetLoader:
    """Get a dataset loader by name.

    Args:
        name: The dataset name.

    Returns:
        An instance of the dataset loader.

    Raises:
        DatasetNotFoundError: If the dataset is not registered.
    """
    if name not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise DatasetNotFoundError(
            f"Dataset '{name}' not found. Available datasets: {available}"
        )

    return _DATASET_REGISTRY[name]()


def list_datasets(category: DatasetCategory | None = None) -> list[DatasetInfo]:
    """List all available datasets.

    Args:
        category: Optional category filter.

    Returns:
        List of dataset info objects.
    """
    datasets = []

    for loader_class in _DATASET_REGISTRY.values():
        instance = loader_class()
        info = instance.info

        if category is None or info.category == category:
            datasets.append(info)

    return datasets


def get_dataset_info(name: str) -> DatasetInfo:
    """Get info about a specific dataset.

    Args:
        name: The dataset name.

    Returns:
        DatasetInfo object.

    Raises:
        DatasetNotFoundError: If the dataset is not registered.
    """
    loader = get_dataset(name)
    return loader.info
