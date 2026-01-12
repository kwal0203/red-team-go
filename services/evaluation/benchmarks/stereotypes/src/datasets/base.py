"""Base classes for stereotype benchmark datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class BiasType(str, Enum):
    """Types of bias measured in stereotype benchmarks."""

    GENDER = "gender"
    RACE = "race"
    RELIGION = "religion"
    AGE = "age"
    NATIONALITY = "nationality"
    DISABILITY = "disability"
    SOCIOECONOMIC = "socioeconomic"
    SEXUAL_ORIENTATION = "sexual_orientation"
    PHYSICAL_APPEARANCE = "physical_appearance"


@dataclass
class BenchmarkItem:
    """Base class for benchmark items.

    Attributes:
        id: Unique identifier for the item.
        bias_type: Type of bias being tested.
    """

    id: str
    bias_type: BiasType

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "bias_type": self.bias_type.value,
        }


@dataclass
class StereoSetItem(BenchmarkItem):
    """Item from the StereoSet benchmark.

    StereoSet tests whether models prefer stereotypical associations.

    Attributes:
        context: The context sentence with a blank.
        stereotype: Stereotypical completion.
        anti_stereotype: Anti-stereotypical completion.
        unrelated: Unrelated completion (baseline).
    """

    context: str
    stereotype: str
    anti_stereotype: str
    unrelated: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "context": self.context,
                "stereotype": self.stereotype,
                "anti_stereotype": self.anti_stereotype,
                "unrelated": self.unrelated,
            }
        )
        return base


@dataclass
class CrowSPairsItem(BenchmarkItem):
    """Item from the CrowS-Pairs benchmark.

    CrowS-Pairs tests model preferences between stereotypical
    and anti-stereotypical sentence pairs.

    Attributes:
        sentence_more_stereotypical: The more stereotypical sentence.
        sentence_less_stereotypical: The less stereotypical sentence.
    """

    sentence_more_stereotypical: str
    sentence_less_stereotypical: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "sentence_more_stereotypical": self.sentence_more_stereotypical,
                "sentence_less_stereotypical": self.sentence_less_stereotypical,
            }
        )
        return base


@dataclass
class BBQItem(BenchmarkItem):
    """Item from the BBQ (Bias Benchmark for QA) dataset.

    BBQ tests whether models use stereotypes to answer ambiguous questions.

    Attributes:
        context: The context/passage.
        question: The question to answer.
        answer_stereotypical: Answer relying on stereotype.
        answer_anti_stereotypical: Answer against stereotype.
        answer_unknown: The correct "unknown" answer for ambiguous cases.
        is_ambiguous: Whether the context is ambiguous.
    """

    context: str
    question: str
    answer_stereotypical: str
    answer_anti_stereotypical: str
    answer_unknown: str
    is_ambiguous: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "context": self.context,
                "question": self.question,
                "answer_stereotypical": self.answer_stereotypical,
                "answer_anti_stereotypical": self.answer_anti_stereotypical,
                "answer_unknown": self.answer_unknown,
                "is_ambiguous": self.is_ambiguous,
            }
        )
        return base


class BaseDataset(ABC):
    """Abstract base class for stereotype benchmark datasets."""

    name: str = "base"
    description: str = "Base dataset"
    citation: str = ""

    @abstractmethod
    def load(self) -> list[BenchmarkItem]:
        """Load the full dataset.

        Returns:
            List of benchmark items.

        Raises:
            NotImplementedError: If dataset loading is not implemented.
        """
        pass

    @abstractmethod
    def get_sample(self, num_samples: int = 10) -> list[BenchmarkItem]:
        """Get a sample of items for testing.

        Args:
            num_samples: Number of samples to return.

        Returns:
            List of sample benchmark items.
        """
        pass

    def filter_by_bias_type(
        self, items: list[BenchmarkItem], bias_types: list[BiasType]
    ) -> list[BenchmarkItem]:
        """Filter items by bias type.

        Args:
            items: List of items to filter.
            bias_types: Bias types to include.

        Returns:
            Filtered list of items.
        """
        return [item for item in items if item.bias_type in bias_types]
