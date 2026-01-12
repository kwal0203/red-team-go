"""Base classes for text perturbations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PerturbedText:
    """Result of a text perturbation.

    Attributes:
        original: The original text before perturbation.
        perturbed: The text after perturbation.
        method: The perturbation method used.
        changes: List of changes made (for explainability).
    """

    original: str
    perturbed: str
    method: str
    changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "method": self.method,
            "changes": self.changes,
        }


class BasePerturbation(ABC):
    """Abstract base class for text perturbations.

    Perturbations modify text in ways that may bypass safety filters
    while preserving semantic meaning.
    """

    name: str = "base"
    category: str = "base"

    @abstractmethod
    def perturb(self, text: str, num_variants: int = 5) -> list[PerturbedText]:
        """Generate perturbed variants of the input text.

        Args:
            text: The original text to perturb.
            num_variants: Number of variants to generate.

        Returns:
            List of PerturbedText objects.
        """
        pass

    def _create_result(
        self,
        original: str,
        perturbed: str,
        method: str,
        changes: list[str] | None = None,
    ) -> PerturbedText:
        """Helper to create a PerturbedText result.

        Args:
            original: Original text.
            perturbed: Perturbed text.
            method: Method name.
            changes: List of changes made.

        Returns:
            PerturbedText object.
        """
        return PerturbedText(
            original=original,
            perturbed=perturbed,
            method=method,
            changes=changes or [],
        )
