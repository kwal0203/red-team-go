"""Base classes for dataset loading infrastructure.

Provides a unified interface for loading various red-teaming datasets
including stereotype benchmarks, jailbreak prompts, and toxicity datasets.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DatasetCategory(str, Enum):
    """Categories of datasets."""

    STEREOTYPE = "stereotype"
    JAILBREAK = "jailbreak"
    TOXICITY = "toxicity"
    HARMFUL = "harmful"
    BIAS = "bias"


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""

    name: str
    category: DatasetCategory
    description: str
    source: str
    citation: str
    size: int | None = None
    huggingface_id: str | None = None


@dataclass(kw_only=True)
class DatasetItem:
    """Base class for dataset items."""

    id: str
    text: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata or {},
        }


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    All dataset loaders should inherit from this class and implement
    the required methods.
    """

    def __init__(self):
        """Initialize the dataset loader."""
        self._data: list[Any] | None = None
        self._loaded = False

    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return dataset metadata."""
        pass

    @abstractmethod
    def load(self) -> list[Any]:
        """Load the full dataset.

        Returns:
            List of dataset items.

        Raises:
            DatasetLoadError: If loading fails.
        """
        pass

    @abstractmethod
    def get_sample(self, num_samples: int = 10) -> list[Any]:
        """Get a sample of the dataset without full loading.

        Useful for testing and demos.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of sample items.
        """
        pass

    def is_loaded(self) -> bool:
        """Check if the dataset is loaded."""
        return self._loaded

    def __len__(self) -> int:
        """Return the number of items in the loaded dataset."""
        if not self._loaded or self._data is None:
            return 0
        return len(self._data)


class DatasetLoadError(Exception):
    """Exception raised when dataset loading fails."""

    pass


class DatasetNotFoundError(Exception):
    """Exception raised when a dataset is not found in the registry."""

    pass
