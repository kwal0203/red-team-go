"""Guardrail pipeline orchestrator.

Manages running multiple guardrails and aggregating results.
"""

import logging
from dataclasses import dataclass

from services.safety.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailResult,
    RiskLevel,
)
from services.safety.guardrails.src.detectors.harmful_content import (
    HarmfulContentDetector,
)
from services.safety.guardrails.src.detectors.injection import InjectionDetector
from services.safety.guardrails.src.detectors.jailbreak import JailbreakDetector
from services.safety.guardrails.src.detectors.privacy import PrivacyDetector
from services.safety.guardrails.src.detectors.toxicity import ToxicityGuardrail

logger = logging.getLogger(__name__)

# Default guardrails registry
DEFAULT_GUARDRAILS = {
    "jailbreak": JailbreakDetector,
    "injection": InjectionDetector,
    "toxicity": ToxicityGuardrail,
    "harmful_content": HarmfulContentDetector,
    "privacy": PrivacyDetector,
}


@dataclass
class PipelineResult:
    """Result from running the guardrail pipeline.

    Attributes:
        results: Dictionary mapping guardrail name to result.
        overall_risk: Aggregated risk level assessment.
        violations: List of guardrail names that detected violations.
        max_confidence: Highest confidence score across all checks.
    """

    results: dict[str, GuardrailResult]
    overall_risk: RiskLevel
    violations: list[str]
    max_confidence: float

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "overall_risk": self.overall_risk.value,
            "violations": self.violations,
            "max_confidence": self.max_confidence,
        }


class GuardrailPipeline:
    """Orchestrates running multiple guardrails on content.

    The pipeline manages:
    - Guardrail registration and selection
    - Running selected guardrails on input/output content
    - Aggregating results and computing overall risk
    """

    def __init__(self, guardrails: list[str] | None = None):
        """Initialize the pipeline with selected guardrails.

        Args:
            guardrails: List of guardrail names to enable. If None or ["all"],
                       enables all available guardrails.
        """
        self._available_guardrails = DEFAULT_GUARDRAILS.copy()
        self._active_guardrails: dict[str, BaseGuardrail] = {}

        # Initialize selected guardrails
        if guardrails is None or guardrails == ["all"] or "all" in guardrails:
            guardrail_names = list(self._available_guardrails.keys())
        else:
            guardrail_names = guardrails

        for name in guardrail_names:
            if name in self._available_guardrails:
                self._active_guardrails[name] = self._available_guardrails[name]()
            else:
                logger.warning(f"Unknown guardrail: {name}")

        logger.info(
            f"Initialized pipeline with {len(self._active_guardrails)} guardrails: "
            f"{list(self._active_guardrails.keys())}"
        )

    def check(self, content: str) -> PipelineResult:
        """Run all active guardrails on content.

        Args:
            content: The text content to check.

        Returns:
            PipelineResult with all guardrail results and overall assessment.
        """
        results: dict[str, GuardrailResult] = {}
        violations: list[str] = []
        max_confidence = 0.0

        for name, guardrail in self._active_guardrails.items():
            try:
                result = guardrail.check(content)
                results[name] = result

                if result.detected:
                    violations.append(name)

                max_confidence = max(max_confidence, result.confidence)

            except Exception as e:
                logger.error(f"Error in guardrail {name}: {e}")
                # Create error result
                results[name] = GuardrailResult(
                    name=name,
                    detected=False,
                    confidence=0.0,
                    explanation=f"Error during check: {str(e)}",
                )

        # Compute overall risk level
        overall_risk = self._compute_risk_level(violations, max_confidence)

        logger.info(
            f"Pipeline check complete: {len(violations)} violations, "
            f"risk={overall_risk.value}, max_confidence={max_confidence:.2f}"
        )

        return PipelineResult(
            results=results,
            overall_risk=overall_risk,
            violations=violations,
            max_confidence=max_confidence,
        )

    def _compute_risk_level(
        self, violations: list[str], max_confidence: float
    ) -> RiskLevel:
        """Compute overall risk level from individual results.

        Args:
            violations: List of guardrails that detected violations.
            max_confidence: Highest confidence across all checks.

        Returns:
            RiskLevel enum value.
        """
        if not violations:
            return RiskLevel.LOW

        # High severity categories
        high_severity = {"harmful_content", "cbrn", "weapons", "violence", "privacy"}
        medium_severity = {"jailbreak", "injection", "toxicity"}

        # Check for critical violations
        if any(v in high_severity for v in violations) and max_confidence >= 0.7:
            return RiskLevel.CRITICAL

        if any(v in high_severity for v in violations) or max_confidence >= 0.7:
            return RiskLevel.HIGH

        if any(v in medium_severity for v in violations) or max_confidence >= 0.4:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    @property
    def available_guardrails(self) -> list[str]:
        """Get list of available guardrail names."""
        return list(self._available_guardrails.keys())

    @property
    def active_guardrails(self) -> list[str]:
        """Get list of active guardrail names."""
        return list(self._active_guardrails.keys())
