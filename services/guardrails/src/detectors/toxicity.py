"""Toxicity detection guardrail.

Wraps the existing toxicity detection service to provide
guardrail-compatible interface.
"""

import logging
import re

from services.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailCategory,
    GuardrailResult,
)

logger = logging.getLogger(__name__)

# Basic toxicity patterns for fast pre-filtering
# These are meant to catch obvious cases before using ML models
TOXICITY_PATTERNS = {
    "profanity": [
        r"\bf+u+c+k+",
        r"\bs+h+[i1]+t+",
        r"\ba+s+s+h+o+l+e+",
        r"\bb+[i1]+t+c+h+",
        r"\bd+a+m+n+",
    ],
    "slurs": [
        r"\bn+[i1]+g+g+",
        r"\bf+a+g+",
        r"\br+e+t+a+r+d+",
        r"\bc+u+n+t+",
    ],
    "threats": [
        r"\b(i.ll |gonna |going to |will )(kill|murder|hurt|harm|destroy)",
        r"\b(you.re |you are )(dead|gonna die)",
        r"\b(death|die|kill)\s+(you|him|her|them)",
        r"\bbeat (you|the|your)",
    ],
    "harassment": [
        r"\b(you|he|she|they) (are|is) (stupid|dumb|idiot|moron)",
        r"\bkill yourself",
        r"\bgo die",
        r"\bnobody (loves|likes|wants) you",
        r"\byou.re (worthless|useless|pathetic)",
    ],
}


class ToxicityGuardrail(BaseGuardrail):
    """Fast toxicity detection using pattern matching.

    For more accurate detection, use the full toxicity detection service
    which uses ML models (OpenAI Moderation + Paradetox).
    """

    def __init__(self, threshold: float = 0.3):
        """Initialize the toxicity guardrail.

        Args:
            threshold: Confidence threshold for detection (0.0-1.0).
        """
        super().__init__(name="toxicity", category=GuardrailCategory.TOXICITY)
        self.threshold = threshold
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in TOXICITY_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check(self, content: str) -> GuardrailResult:
        """Check content for toxic language.

        This performs fast pattern-based detection. For production use,
        consider using the full ML-based toxicity detection service.

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
                    # Weight different categories
                    if category == "slurs":
                        total_score += 0.5
                    elif category == "threats":
                        total_score += 0.4
                    elif category == "harassment" or category == "profanity":
                        total_score += 0.35
                    else:
                        total_score += 0.3

        confidence = min(1.0, total_score)

        if not matches:
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="No toxic patterns detected",
            )

        # Determine primary category
        category_counts: dict[str, int] = {}
        for cat, _ in matches:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        # Generate explanation
        explanations = {
            "profanity": "Detected profane language",
            "slurs": "Detected discriminatory slurs",
            "threats": "Detected threatening language",
            "harassment": "Detected harassing content",
        }
        explanation = explanations.get(primary_category, "Detected toxic content")

        logger.debug(
            f"Toxicity detection: {len(matches)} patterns matched, "
            f"confidence={confidence:.2f}, category={primary_category}"
        )

        return self._create_result(
            detected=confidence >= self.threshold,
            confidence=confidence,
            explanation=explanation,
            category=primary_category,
        )
