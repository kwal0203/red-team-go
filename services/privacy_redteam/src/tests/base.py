"""Base class for privacy red team tests.

Provides common functionality for all privacy tests including model interaction,
leakage detection, and result formatting.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from services.privacy_redteam.src.analyzer import (
    LeakageAnalyzer,
    LeakageResult,
    LeakageType,
)

logger = logging.getLogger(__name__)


@dataclass
class PrivacySample:
    """A single privacy test sample with results."""

    probe_name: str
    probe_category: str
    prompt: str
    model_response: str
    leakage_result: LeakageResult
    is_safe: bool
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert sample to dictionary.

        Returns:
            Dictionary representation of the sample.
        """
        return {
            "probe_name": self.probe_name,
            "probe_category": self.probe_category,
            "prompt": self.prompt,
            "model_response": self.model_response[:500] + "..."
            if len(self.model_response) > 500
            else self.model_response,
            "leakage_result": self.leakage_result.to_dict(),
            "is_safe": self.is_safe,
            "metadata": self.metadata,
        }


class BasePrivacyTest(ABC):
    """Abstract base class for privacy red team tests.

    Subclasses must implement the run() method which executes the
    specific test logic and returns results.
    """

    name: str = "base"
    description: str = "Base privacy test"

    def __init__(
        self,
        num_samples: int = 5,
        system_prompt: str | None = None,
        **kwargs,
    ):
        """Initialize the base privacy test.

        Args:
            num_samples: Number of probes to run per category.
            system_prompt: Optional system prompt for testing leakage.
            **kwargs: Additional configuration options.
        """
        self.num_samples = num_samples
        self.system_prompt = system_prompt
        self.config = kwargs
        self.analyzer = LeakageAnalyzer()

    @abstractmethod
    def run(self, model) -> dict:
        """Run the privacy test.

        Args:
            model: Target model wrapper with model_predict() method.

        Returns:
            Dictionary with score, details, and samples.
        """
        pass

    def _get_model_response(self, model, prompt: str) -> str:
        """Get a response from the target model.

        Args:
            model: Target model wrapper.
            prompt: The prompt to send.

        Returns:
            Model's response text.
        """
        try:
            response = model.model_predict(prompt)
            # model_predict returns a list, get first element
            if isinstance(response, list):
                return response[0] if response else ""
            return response if response else ""
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return ""

    def _get_model_response_with_system(
        self, model, prompt: str, system_prompt: str
    ) -> str:
        """Get a response from the model with a system prompt.

        Args:
            model: Target model wrapper.
            prompt: The user prompt to send.
            system_prompt: The system prompt to use.

        Returns:
            Model's response text.
        """
        try:
            # Prepend system prompt to user prompt
            combined = f"System: {system_prompt}\n\nUser: {prompt}"
            response = model.model_predict(combined)
            # model_predict returns a list, get first element
            if isinstance(response, list):
                return response[0] if response else ""
            return response if response else ""
        except Exception as e:
            logger.error(f"Error getting model response with system prompt: {e}")
            return ""

    def _analyze_response(
        self,
        response: str,
        check_pii: bool = True,
        check_instructions: bool = True,
        check_confidence: bool = True,
        test_strings: list[str] | None = None,
    ) -> LeakageResult:
        """Analyze a response for privacy leakage.

        Args:
            response: The model's response text.
            check_pii: Whether to check for PII patterns.
            check_instructions: Whether to check for instruction leakage.
            check_confidence: Whether to check for inappropriate confidence.
            test_strings: Optional strings to check for verbatim leakage.

        Returns:
            LeakageResult with detection details.
        """
        return self.analyzer.analyze(
            response=response,
            check_pii=check_pii,
            check_instructions=check_instructions,
            check_confidence=check_confidence,
            test_strings=test_strings,
            system_prompt=self.system_prompt,
        )

    def _calculate_safety_score(self, samples: list[PrivacySample]) -> float:
        """Calculate safety score from samples.

        Higher score = better (fewer leakages).

        Args:
            samples: List of test samples.

        Returns:
            Safety score between 0.0 and 1.0.
        """
        if not samples:
            return 1.0

        safe_count = sum(1 for s in samples if s.is_safe)
        return safe_count / len(samples)

    def _get_leakage_breakdown(self, samples: list[PrivacySample]) -> dict[str, int]:
        """Get count of each leakage type in samples.

        Args:
            samples: List of test samples.

        Returns:
            Dictionary with counts by LeakageType.
        """
        breakdown = {lt.value: 0 for lt in LeakageType}

        for sample in samples:
            leakage_type = sample.leakage_result.leakage_type
            breakdown[leakage_type.value] += 1

        return breakdown

    def _get_category_breakdown(
        self, samples: list[PrivacySample]
    ) -> dict[str, dict[str, int]]:
        """Get safety breakdown by probe category.

        Args:
            samples: List of test samples.

        Returns:
            Dictionary mapping categories to safe/leaked counts.
        """
        breakdown: dict[str, dict[str, int]] = {}

        for sample in samples:
            category = sample.probe_category
            if category not in breakdown:
                breakdown[category] = {"safe": 0, "leaked": 0}

            if sample.is_safe:
                breakdown[category]["safe"] += 1
            else:
                breakdown[category]["leaked"] += 1

        return breakdown

    def _score_to_dict(
        self,
        score: float,
        samples: list[PrivacySample],
        extra_details: dict | None = None,
    ) -> dict:
        """Format test results as a dictionary.

        Args:
            score: Overall safety score (0.0 to 1.0).
            samples: List of test samples.
            extra_details: Additional details to include.

        Returns:
            Formatted result dictionary.
        """
        leakage_breakdown = self._get_leakage_breakdown(samples)
        category_breakdown = self._get_category_breakdown(samples)

        # Count total leakages
        total_leakages = sum(
            count
            for lt, count in leakage_breakdown.items()
            if lt != LeakageType.NO_LEAKAGE.value
        )

        details = {
            "leakage_breakdown": leakage_breakdown,
            "category_breakdown": category_breakdown,
            "total_samples": len(samples),
            "total_leakages": total_leakages,
            "leakage_rate": total_leakages / len(samples) if samples else 0.0,
        }

        if extra_details:
            details.update(extra_details)

        return {
            "score": round(score, 3),
            "details": details,
            "samples": [s.to_dict() for s in samples],
        }

    def _create_sample(
        self,
        probe_name: str,
        probe_category: str,
        prompt: str,
        response: str,
        leakage_result: LeakageResult,
        metadata: dict | None = None,
    ) -> PrivacySample:
        """Create a privacy sample from test results.

        Args:
            probe_name: Name of the probe used.
            probe_category: Category of the probe.
            prompt: The prompt that was sent.
            response: The model's response.
            leakage_result: Result from leakage analysis.
            metadata: Additional metadata.

        Returns:
            PrivacySample instance.
        """
        return PrivacySample(
            probe_name=probe_name,
            probe_category=probe_category,
            prompt=prompt,
            model_response=response,
            leakage_result=leakage_result,
            is_safe=not leakage_result.detected,
            metadata=metadata or {},
        )
