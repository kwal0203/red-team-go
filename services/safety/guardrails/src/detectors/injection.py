"""Prompt injection detection guardrail.

Detects attempts to inject malicious instructions into prompts,
including direct injections and indirect injections via documents/URLs.
"""

import logging
import re

from services.safety.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailCategory,
    GuardrailResult,
)

logger = logging.getLogger(__name__)

# Pattern categories for prompt injection detection
INJECTION_PATTERNS = {
    "instruction_override": [
        r"ignore (the )?(above|previous|prior|all)",
        r"disregard (the )?(above|previous|prior|all)",
        r"forget (the )?(above|previous|prior|all)",
        r"do not follow (the )?(above|previous|prior)",
        r"instead[,\s]+(do|say|respond|output)",
        r"new instructions?:",
        r"updated instructions?:",
        r"real instructions?:",
        r"actual task:",
        r"actual instructions?:",
    ],
    "context_manipulation": [
        r"---\s*(new|actual|real)\s*(instructions?|task|prompt)",
        r"=+\s*(new|actual|real)\s*(instructions?|task)",
        r"\[hidden\]",
        r"\[secret\]",
        r"\[system\s*override\]",
        r"<!--.*-->",  # HTML comments
        r"\{\{.*\}\}",  # Template injection
        r"\$\{.*\}",  # Variable injection
    ],
    "role_hijacking": [
        r"you are (now )?a (different|new|helpful)",
        r"your (new )?role is",
        r"switch to (a )?new (role|persona|character)",
        r"adopt (the )?(role|persona) of",
        r"respond as (a |an )?(?!human)",
        r"act as (an? )?assistant (that|who)",
    ],
    "output_manipulation": [
        r"output only",
        r"respond (only )?with",
        r"say (only )?\"",
        r"print (only )?:",
        r"return (only )?:",
        r"your (only )?response (should|must) be",
        r"format (your )?(response|output) as",
    ],
    "delimiter_injection": [
        r"```system",
        r"```assistant",
        r"```user",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"### (System|User|Assistant):",
        r"<s>\[INST\]",
        r"\[/INST\]",
    ],
}

# Suspicious structural patterns
STRUCTURAL_INDICATORS = [
    (r"---+", 0.1),  # Multiple dashes (context separator)
    (r"===+", 0.1),  # Multiple equals
    (r"\n\n\n+", 0.05),  # Excessive newlines
    (r"[A-Z]{5,}", 0.05),  # All caps text (emphasis)
]


class InjectionDetector(BaseGuardrail):
    """Detects prompt injection attempts.

    Uses pattern matching to identify various prompt injection techniques:
    - Direct instruction overrides
    - Context/delimiter manipulation
    - Role hijacking
    - Output format manipulation
    """

    def __init__(self):
        """Initialize the injection detector."""
        super().__init__(name="injection", category=GuardrailCategory.INJECTION)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in INJECTION_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._structural_patterns = [
            (re.compile(p), score) for p, score in STRUCTURAL_INDICATORS
        ]

    def check(self, content: str) -> GuardrailResult:
        """Check content for prompt injection attempts.

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
                    match_str = pattern.pattern
                    matches.append((category, match_str))
                    # Weight different categories
                    if category == "instruction_override":
                        total_score += 0.35
                    elif category == "delimiter_injection":
                        total_score += 0.4
                    elif (
                        category == "context_manipulation"
                        or category == "role_hijacking"
                    ):
                        total_score += 0.3
                    elif category == "output_manipulation":
                        total_score += 0.35
                    else:
                        total_score += 0.25

        # Check structural indicators (lower weight)
        for pattern, base_score in self._structural_patterns:
            if pattern.search(content):
                # Count occurrences and add diminishing score
                count = len(pattern.findall(content))
                total_score += base_score * min(count, 3)

        # Calculate confidence (cap at 1.0)
        confidence = min(1.0, total_score)

        if not matches:
            return self._create_result(
                detected=False,
                confidence=confidence,
                explanation="No prompt injection patterns detected",
            )

        # Determine primary category
        category_counts: dict[str, int] = {}
        for cat, _ in matches:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        # Generate explanation
        explanations = {
            "instruction_override": "Detected instruction override attempt",
            "context_manipulation": "Detected context manipulation attempt",
            "role_hijacking": "Detected role hijacking attempt",
            "output_manipulation": "Detected output manipulation attempt",
            "delimiter_injection": "Detected delimiter/format injection attempt",
        }
        explanation = explanations.get(
            primary_category, "Detected prompt injection attempt"
        )

        logger.debug(
            f"Injection detection: {len(matches)} patterns matched, "
            f"confidence={confidence:.2f}, category={primary_category}"
        )

        return self._create_result(
            detected=confidence >= 0.3,  # Threshold for detection
            confidence=confidence,
            explanation=explanation,
            category=primary_category,
        )
