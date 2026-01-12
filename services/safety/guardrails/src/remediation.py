"""Output remediation logic for protect mode.

Provides functionality to redact, filter, or modify content
that has been flagged by guardrails.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from services.safety.guardrails.src.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class RemediationAction(str, Enum):
    """Actions that can be taken on flagged content."""

    BLOCK = "block"  # Block content entirely
    FLAG = "flag"  # Allow but flag for review
    REDACT = "redact"  # Attempt to redact harmful parts


@dataclass
class RemediationResult:
    """Result of remediation attempt.

    Attributes:
        allowed: Whether content should be allowed through.
        action_taken: The remediation action that was applied.
        original_content: The original content before remediation.
        remediated_content: Modified content (if redaction applied).
        violations: List of detected violations.
        explanation: Description of what was done.
    """

    allowed: bool
    action_taken: RemediationAction
    original_content: str
    remediated_content: str | None
    violations: list[str]
    explanation: str

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "allowed": self.allowed,
            "action_taken": self.action_taken.value,
            "original_content": self.original_content,
            "remediated_content": self.remediated_content,
            "violations": self.violations,
            "explanation": self.explanation,
        }


# Patterns for redaction
REDACTION_PATTERNS = {
    "profanity": [
        (r"\bf+u+c+k+\w*", "[REDACTED]"),
        (r"\bs+h+[i1]+t+\w*", "[REDACTED]"),
        (r"\ba+s+s+h+o+l+e+\w*", "[REDACTED]"),
        (r"\bb+[i1]+t+c+h+\w*", "[REDACTED]"),
    ],
    "slurs": [
        (r"\bn+[i1]+g+g+\w*", "[REDACTED]"),
        (r"\bf+a+g+\w*", "[REDACTED]"),
        (r"\br+e+t+a+r+d+\w*", "[REDACTED]"),
    ],
    "sensitive_info": [
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
        # Phone numbers (various US formats)
        (
            r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
            "[PHONE REDACTED]",
        ),
        # SSN
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
        # Credit card (16-digit and Amex formats)
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD REDACTED]"),
        (r"\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b", "[CARD REDACTED]"),
        # IP addresses
        (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "[IP REDACTED]",
        ),
        # API keys and secrets
        (r"\bsk-[a-zA-Z0-9]{32,}\b", "[API KEY REDACTED]"),
        (r"\bghp_[a-zA-Z0-9]{36}\b", "[API KEY REDACTED]"),
        (r"\b(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b", "[API KEY REDACTED]"),
    ],
}


class ContentRemediator:
    """Handles content remediation for protect mode.

    Supports three actions:
    - BLOCK: Reject content entirely
    - FLAG: Allow content but mark for review
    - REDACT: Attempt to remove harmful parts
    """

    def __init__(self, default_action: RemediationAction = RemediationAction.REDACT):
        """Initialize the remediator.

        Args:
            default_action: Default action to take when violations detected.
        """
        self.default_action = default_action
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile redaction patterns."""
        self._redaction_patterns: list[tuple[re.Pattern, str]] = []
        for patterns in REDACTION_PATTERNS.values():
            for pattern, replacement in patterns:
                self._redaction_patterns.append(
                    (re.compile(pattern, re.IGNORECASE), replacement)
                )

    def remediate(
        self,
        content: str,
        pipeline_result: PipelineResult,
        action: RemediationAction | None = None,
    ) -> RemediationResult:
        """Apply remediation based on guardrail results.

        Args:
            content: The content to remediate.
            pipeline_result: Results from the guardrail pipeline.
            action: Override action to take. If None, uses default.

        Returns:
            RemediationResult with outcome and any modifications.
        """
        action = action or self.default_action
        violations = pipeline_result.violations

        # No violations - allow through
        if not violations:
            return RemediationResult(
                allowed=True,
                action_taken=RemediationAction.FLAG,
                original_content=content,
                remediated_content=None,
                violations=[],
                explanation="No violations detected, content allowed",
            )

        # Handle based on action type
        if action == RemediationAction.BLOCK:
            return self._block_content(content, violations)
        elif action == RemediationAction.FLAG:
            return self._flag_content(content, violations)
        else:  # REDACT
            return self._redact_content(content, violations, pipeline_result)

    def _block_content(self, content: str, violations: list[str]) -> RemediationResult:
        """Block content entirely."""
        return RemediationResult(
            allowed=False,
            action_taken=RemediationAction.BLOCK,
            original_content=content,
            remediated_content=None,
            violations=violations,
            explanation=f"Content blocked due to violations: {', '.join(violations)}",
        )

    def _flag_content(self, content: str, violations: list[str]) -> RemediationResult:
        """Allow content but flag for review."""
        return RemediationResult(
            allowed=True,
            action_taken=RemediationAction.FLAG,
            original_content=content,
            remediated_content=None,
            violations=violations,
            explanation=f"Content flagged for review: {', '.join(violations)}",
        )

    def _redact_content(
        self,
        content: str,
        violations: list[str],
        pipeline_result: PipelineResult,
    ) -> RemediationResult:
        """Attempt to redact harmful parts of content."""
        # Check if we can redact or should block
        # Critical violations cannot be redacted
        if pipeline_result.overall_risk.value == "critical":
            logger.info("Critical risk detected, blocking instead of redacting")
            return self._block_content(content, violations)

        # Apply redaction patterns
        redacted_content = content
        redactions_made = 0

        for pattern, replacement in self._redaction_patterns:
            new_content = pattern.sub(replacement, redacted_content)
            if new_content != redacted_content:
                redactions_made += 1
                redacted_content = new_content

        # If we made redactions, allow the modified content
        if redactions_made > 0:
            logger.info(f"Made {redactions_made} redactions to content")
            return RemediationResult(
                allowed=True,
                action_taken=RemediationAction.REDACT,
                original_content=content,
                remediated_content=redacted_content,
                violations=violations,
                explanation=f"Content modified: {redactions_made} redactions applied",
            )

        # Couldn't redact, fall back to flagging
        logger.info("No redactions applicable, flagging content")
        return RemediationResult(
            allowed=True,
            action_taken=RemediationAction.FLAG,
            original_content=content,
            remediated_content=None,
            violations=violations,
            explanation="Could not redact, content flagged for review",
        )
