"""Base guardrail interface and result types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class GuardrailCategory(str, Enum):
    """Categories of guardrail checks."""

    JAILBREAK = "jailbreak"
    INJECTION = "injection"
    TOXICITY = "toxicity"
    HARMFUL_CONTENT = "harmful_content"


class RiskLevel(str, Enum):
    """Overall risk level assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result from a guardrail check.

    Attributes:
        name: The guardrail name (e.g., "jailbreak", "injection").
        detected: Whether a violation was detected.
        confidence: Confidence score from 0.0 to 1.0.
        explanation: Short description of why content was flagged.
        category: Sub-category if applicable (e.g., "roleplay_attack").
    """

    name: str
    detected: bool
    confidence: float
    explanation: str
    category: str | None = None

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "detected": self.detected,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "category": self.category,
        }


class BaseGuardrail(ABC):
    """Abstract base class for all guardrail detectors.

    Each guardrail must implement the check() method to analyze content
    and return a GuardrailResult.
    """

    def __init__(self, name: str, category: GuardrailCategory):
        """Initialize the guardrail.

        Args:
            name: Unique identifier for this guardrail.
            category: The category of check this guardrail performs.
        """
        self.name = name
        self.category = category

    @abstractmethod
    def check(self, content: str) -> GuardrailResult:
        """Check content for violations.

        Args:
            content: The text content to analyze.

        Returns:
            GuardrailResult with detection status and explanation.
        """
        pass

    def _create_result(
        self,
        detected: bool,
        confidence: float,
        explanation: str,
        category: str | None = None,
    ) -> GuardrailResult:
        """Helper to create a standardized result.

        Args:
            detected: Whether a violation was detected.
            confidence: Confidence score (0.0-1.0).
            explanation: Description of the finding.
            category: Optional sub-category.

        Returns:
            Formatted GuardrailResult.
        """
        return GuardrailResult(
            name=self.name,
            detected=detected,
            confidence=min(1.0, max(0.0, confidence)),
            explanation=explanation,
            category=category,
        )
