"""Jailbreak detection guardrail.

Detects attempts to bypass LLM safety guidelines through various techniques
including DAN-style prompts, roleplay attacks, and trust-building patterns.
"""

import logging
import re

from services.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailCategory,
    GuardrailResult,
)

logger = logging.getLogger(__name__)

# Pattern categories for jailbreak detection
JAILBREAK_PATTERNS = {
    "dan_style": [
        r"\bDAN\b",
        r"do anything now",
        r"jailbreak(ed)?",
        r"ignore (all )?(previous |prior |your )?(instructions|rules|guidelines)",
        r"pretend (you are|to be|that you)",
        r"act as (if|though)",
        r"you are now",
        r"from now on",
        r"new persona",
        r"developer mode",
        r"sudo mode",
        r"admin mode",
        r"god mode",
        r"unrestricted mode",
    ],
    "roleplay_attack": [
        r"you are (a |an )?(evil|malicious|unethical)",
        r"play the role of",
        r"roleplay as",
        r"imagine you (are|have no)",
        r"in this hypothetical",
        r"in a fictional (world|scenario)",
        r"as a character who",
        r"pretend there are no (rules|limits|restrictions)",
        r"you have no (ethics|morals|restrictions|limits)",
        r"you don.t have (to follow|any) (rules|guidelines)",
    ],
    "system_override": [
        r"\[system\]",
        r"\[SYSTEM\]",
        r"system prompt:",
        r"override (safety|content) (policy|filter)",
        r"bypass (safety|content|security)",
        r"disable (safety|filter|moderation)",
        r"turn off (safety|filter)",
        r"ignore (safety|ethical|moral) (guidelines|constraints)",
    ],
    "encoding_evasion": [
        r"base64[:\s]",
        r"encode(d)? (in|as|using)",
        r"decode this",
        r"rot13",
        r"caesar cipher",
        r"pig latin",
        r"backwards:",
        r"reverse this",
    ],
}

# Phrases that indicate attempts to manipulate the model
MANIPULATION_PHRASES = [
    "you must comply",
    "you have to answer",
    "you cannot refuse",
    "you are required to",
    "answer without restrictions",
    "respond without limitations",
    "no content policy",
    "no safety guidelines",
    "forget your training",
    "disregard your programming",
]


class JailbreakDetector(BaseGuardrail):
    """Detects jailbreak attempts in prompts.

    Uses a combination of pattern matching and heuristics to identify
    various jailbreak techniques including:
    - DAN (Do Anything Now) style prompts
    - Roleplay-based attacks
    - System override attempts
    - Encoding/obfuscation evasion
    """

    def __init__(self):
        """Initialize the jailbreak detector."""
        super().__init__(name="jailbreak", category=GuardrailCategory.JAILBREAK)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in JAILBREAK_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def check(self, content: str) -> GuardrailResult:
        """Check content for jailbreak attempts.

        Args:
            content: The text to analyze (typically input prompt).

        Returns:
            GuardrailResult with detection status and explanation.
        """
        if not content or not content.strip():
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="Empty content",
            )

        content_lower = content.lower()
        matches: list[tuple[str, str]] = []
        total_score = 0.0

        # Check compiled patterns
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    match_str = pattern.pattern
                    matches.append((category, match_str))
                    # Weight different categories differently
                    if category == "system_override":
                        total_score += 0.4
                    elif category == "dan_style":
                        total_score += 0.3
                    elif category == "roleplay_attack":
                        total_score += 0.25
                    else:
                        total_score += 0.2

        # Check manipulation phrases
        for phrase in MANIPULATION_PHRASES:
            if phrase in content_lower:
                matches.append(("manipulation", phrase))
                total_score += 0.2

        # Calculate confidence (cap at 1.0)
        confidence = min(1.0, total_score)

        if not matches:
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="No jailbreak patterns detected",
            )

        # Determine primary category
        category_counts: dict[str, int] = {}
        for cat, _ in matches:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        # Generate explanation
        if primary_category == "dan_style":
            explanation = "Detected DAN-style jailbreak attempt"
        elif primary_category == "roleplay_attack":
            explanation = "Detected roleplay-based jailbreak attempt"
        elif primary_category == "system_override":
            explanation = "Detected system override attempt"
        elif primary_category == "encoding_evasion":
            explanation = "Detected encoding-based evasion attempt"
        else:
            explanation = "Detected manipulation phrases attempting to bypass safety"

        logger.debug(
            f"Jailbreak detection: {len(matches)} patterns matched, "
            f"confidence={confidence:.2f}, category={primary_category}"
        )

        return self._create_result(
            detected=confidence >= 0.3,  # Threshold for detection
            confidence=confidence,
            explanation=explanation,
            category=primary_category,
        )
