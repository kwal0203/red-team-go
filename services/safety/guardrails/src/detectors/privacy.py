"""Privacy detection guardrail for PII leakage.

Implements Privacy Red Teaming (Item 5.5) for detecting:
- Email addresses
- Phone numbers (US formats)
- Social Security Numbers
- Credit card numbers
- IP addresses
- Home/street addresses
- API keys and secrets
"""

import logging
import re

from services.safety.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailCategory,
    GuardrailResult,
)

logger = logging.getLogger(__name__)

# Pattern categories for PII detection
PII_PATTERNS = {
    "email": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ],
    "phone": [
        # US formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
        r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    ],
    "ssn": [
        # SSN with dashes: 123-45-6789
        r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
    ],
    "credit_card": [
        # Standard 16-digit cards: 1234 5678 9012 3456 or 1234-5678-9012-3456
        r"\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b",
        # Amex format: 1234 567890 12345
        r"\b[0-9]{4}[-\s]?[0-9]{6}[-\s]?[0-9]{5}\b",
    ],
    "ip_address": [
        # IPv4 addresses with proper octet validation
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    ],
    "api_key": [
        # Generic API key patterns: api_key=xxx, token: xxx, etc.
        r"(?i)(api[_-]?key|apikey|secret[_-]?key|auth[_-]?token|access[_-]?token|password)\s*[=:]\s*['\"]?[\w-]{16,}['\"]?",
        # Bearer/Basic auth tokens
        r"(?i)(bearer|basic)\s+[A-Za-z0-9_-]{20,}",
        # OpenAI API key format
        r"\bsk-[a-zA-Z0-9]{32,}\b",
        # GitHub Personal Access Token
        r"\bghp_[a-zA-Z0-9]{36}\b",
        # AWS access key
        r"\b(AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b",
    ],
    "address": [
        # Street addresses: 123 Main Street, 456 Oak Ave, etc.
        r"\b\d+\s+[A-Za-z]+(\s+[A-Za-z]+)?\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct|place|pl|circle|cir)\b",
    ],
}

# Risk weights for different PII types
PII_RISK_WEIGHTS = {
    "ssn": 0.6,  # Critical - full identity theft risk
    "credit_card": 0.6,  # Critical - financial risk
    "api_key": 0.5,  # High - security breach risk
    "email": 0.4,  # Medium - privacy concern
    "phone": 0.4,  # Medium - privacy concern
    "address": 0.35,  # Medium - privacy concern
    "ip_address": 0.3,  # Lower - but still concerning
}


class PrivacyDetector(BaseGuardrail):
    """Detects personally identifiable information (PII) leakage.

    Implements PII detection for:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Street addresses
    - API keys and secrets
    """

    def __init__(self, threshold: float = 0.3):
        """Initialize the privacy detector.

        Args:
            threshold: Confidence threshold for detection (0.0-1.0).
        """
        super().__init__(name="privacy", category=GuardrailCategory.PRIVACY)
        self.threshold = threshold
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in PII_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE if category != "api_key" else 0)
                for p in patterns
            ]

    def check(self, content: str) -> GuardrailResult:
        """Check content for PII leakage.

        Args:
            content: The text to analyze.

        Returns:
            GuardrailResult with detection status and explanation.
        """
        if not content or not content.strip():
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="Empty content",
            )

        matches: list[tuple[str, str]] = []
        total_score = 0.0

        # Check compiled patterns
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    matches.append((category, pattern.pattern))
                    # Apply category-specific risk weight
                    weight = PII_RISK_WEIGHTS.get(category, 0.3)
                    total_score += weight

        confidence = min(1.0, total_score)

        if not matches:
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="No PII patterns detected",
            )

        # Determine primary category (most matches)
        category_counts: dict[str, int] = {}
        for cat, _ in matches:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        # Generate explanation
        explanations = {
            "email": "Detected email address",
            "phone": "Detected phone number",
            "ssn": "Detected Social Security Number",
            "credit_card": "Detected credit card number",
            "ip_address": "Detected IP address",
            "api_key": "Detected API key or secret token",
            "address": "Detected street address",
        }
        explanation = explanations.get(primary_category, "Detected PII leakage")

        # Add count if multiple types detected
        if len(category_counts) > 1:
            pii_types = ", ".join(sorted(category_counts.keys()))
            explanation = f"Detected multiple PII types: {pii_types}"

        logger.debug(
            f"Privacy detection: {len(matches)} patterns matched, "
            f"confidence={confidence:.2f}, categories={list(category_counts.keys())}"
        )

        return self._create_result(
            detected=confidence >= self.threshold,
            confidence=confidence,
            explanation=explanation,
            category=primary_category,
        )
