"""Dataset loading infrastructure for red-teaming benchmarks.

Provides loaders for various datasets including stereotype benchmarks,
jailbreak prompts, and toxicity datasets.
"""

# Import loaders to trigger registration
from services.shared.datasets.src import stereotype  # noqa: F401
from services.shared.datasets.src.base import (
    BaseDatasetLoader,
    DatasetCategory,
    DatasetInfo,
    DatasetItem,
    DatasetLoadError,
    DatasetNotFoundError,
)
from services.shared.datasets.src.registry import (
    get_dataset,
    get_dataset_info,
    list_datasets,
    register_dataset,
)

__all__ = [
    # Base classes
    "BaseDatasetLoader",
    "DatasetCategory",
    "DatasetInfo",
    "DatasetItem",
    "DatasetLoadError",
    "DatasetNotFoundError",
    # Registry functions
    "get_dataset",
    "get_dataset_info",
    "list_datasets",
    "register_dataset",
]
