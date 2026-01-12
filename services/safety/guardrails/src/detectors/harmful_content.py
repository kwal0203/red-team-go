"""Harmful content detection guardrail.

Implements safety evaluation (Phase 2, Item 6) for detecting:
- Harmful instructions (weapons, illegal activities)
- Dangerous content (self-harm, violence)
- Dual-use content (security research that could enable attacks)
- CBRN (Chemical, Biological, Radiological, Nuclear) content
"""

import logging
import re

from services.safety.guardrails.src.detectors.base import (
    BaseGuardrail,
    GuardrailCategory,
    GuardrailResult,
)

logger = logging.getLogger(__name__)

# Pattern categories for harmful content detection
HARMFUL_PATTERNS = {
    "weapons": [
        r"\b(how to |make |build |create |construct )(a |an )?(bomb|explosive|weapon|gun|firearm)",
        r"\b(homemade |diy |improvised )(explosive|weapon|bomb)",
        r"\b(3d print|manufacture) (a |an )?(gun|firearm|weapon)",
        r"\bassembly (of |for )(weapons?|firearms?)",
        r"\bweapon(ize|ization)",
    ],
    "illegal_activities": [
        r"\b(how to |ways to )(hack|steal|launder|counterfeit|forge)",
        r"\b(bypass|evade) (security|detection|law enforcement)",
        r"\b(illegal |illicit )(drug|substance) (synthesis|production|manufacturing)",
        r"\b(identity|credit card) (theft|fraud|scam)",
        r"\bpick (a )?lock",
        r"\bhot.?wire (a )?car",
    ],
    "self_harm": [
        r"\b(how to |ways to |methods (for|of) )(kill|harm|hurt) (yourself|myself|oneself)",
        r"\bsuicide (methods?|techniques?|instructions?)",
        r"\b(end|take) (your|my|one.s) (own )?life",
        r"\bself.?(harm|mutilation|injury)",
        r"\bcutting (yourself|myself)",
    ],
    "violence": [
        r"\b(how to |ways to )(murder|assassinate|kill) (someone|a person|people)",
        r"\b(torture|mutilate) (a |an )?(person|human|victim)",
        r"\bmass (shooting|casualty|violence)",
        r"\bschool (shooting|attack)",
        r"\bterrorist (attack|act|plot)",
    ],
    "cbrn": [
        r"\b(synthesize|make|produce) (poison|toxin|venom)",
        r"\b(chemical|biological) (weapon|agent|warfare)",
        r"\b(nerve (gas|agent)|mustard gas|sarin|vx)",
        r"\b(ricin|anthrax|botulinum)",
        r"\b(dirty bomb|radioactive|nuclear) (weapon|device|bomb)",
        r"\buranium (enrichment|procurement)",
    ],
    "dual_use": [
        r"\b(exploit|vulnerability|zero.?day) (code|script|program)",
        r"\b(malware|ransomware|trojan|keylogger) (code|source|development)",
        r"\b(bypass|evade|disable) (antivirus|security|firewall|detection)",
        r"\bphishing (kit|template|page)",
        r"\b(sql injection|xss|rce) (attack|payload|exploit)",
        r"\bcrack(ing)? (password|encryption|hash)",
    ],
}

# High-risk indicators that increase confidence
HIGH_RISK_INDICATORS = [
    (r"\bstep.?by.?step", 0.15),
    (r"\bdetailed (instructions?|guide)", 0.15),
    (r"\bworking (code|script|exploit)", 0.2),
    (r"\btested (and )?(working|verified)", 0.1),
    (r"\bundetectable", 0.15),
    (r"\buntraceable", 0.15),
]


class HarmfulContentDetector(BaseGuardrail):
    """Detects harmful and dangerous content.

    Implements comprehensive safety evaluation for:
    - Weapons and explosives
    - Illegal activities
    - Self-harm content
    - Violence
    - CBRN (Chemical, Biological, Radiological, Nuclear)
    - Dual-use security content
    """

    def __init__(self, threshold: float = 0.3):
        """Initialize the harmful content detector.

        Args:
            threshold: Confidence threshold for detection (0.0-1.0).
        """
        super().__init__(
            name="harmful_content", category=GuardrailCategory.HARMFUL_CONTENT
        )
        self.threshold = threshold
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in HARMFUL_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._risk_indicators = [
            (re.compile(p, re.IGNORECASE), score) for p, score in HIGH_RISK_INDICATORS
        ]

    def check(self, content: str) -> GuardrailResult:
        """Check content for harmful material.

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
                    # Weight different categories by severity
                    if category == "cbrn":
                        total_score += 0.6
                    elif (
                        category == "weapons"
                        or category == "violence"
                        or category == "self_harm"
                    ):
                        total_score += 0.5
                    elif category == "illegal_activities":
                        total_score += 0.4
                    elif category == "dual_use":
                        total_score += 0.3

        # Check for high-risk indicators
        for pattern, score in self._risk_indicators:
            if pattern.search(content):
                total_score += score

        confidence = min(1.0, total_score)

        if not matches:
            return self._create_result(
                detected=False,
                confidence=0.0,
                explanation="No harmful content patterns detected",
            )

        # Determine primary category
        category_counts: dict[str, int] = {}
        for cat, _ in matches:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        primary_category = max(category_counts, key=lambda k: category_counts[k])

        # Generate explanation
        explanations = {
            "weapons": "Detected weapons/explosives related content",
            "illegal_activities": "Detected content related to illegal activities",
            "self_harm": "Detected self-harm related content",
            "violence": "Detected violent/harmful content",
            "cbrn": "Detected CBRN (Chemical/Biological/Radiological/Nuclear) content",
            "dual_use": "Detected dual-use security content that could enable attacks",
        }
        explanation = explanations.get(
            primary_category, "Detected potentially harmful content"
        )

        logger.debug(
            f"Harmful content detection: {len(matches)} patterns matched, "
            f"confidence={confidence:.2f}, category={primary_category}"
        )

        return self._create_result(
            detected=confidence >= self.threshold,
            confidence=confidence,
            explanation=explanation,
            category=primary_category,
        )
