"""Confidence calculation methods for hallucination detection."""

import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceMethod(str, Enum):
    """Available confidence calculation methods."""

    AVERAGE = "average"  # Average of token probabilities
    GEOMETRIC = "geometric"  # Geometric mean (sequence probability)
    MINIMUM = "minimum"  # Minimum token probability (pessimistic)
    ENTROPY = "entropy"  # Entropy-based uncertainty measure
    VARIANCE = "variance"  # Variance-based consistency measure


@dataclass
class ConfidenceResult:
    """Result from confidence calculation."""

    score: float  # 0-100 scale
    method: ConfidenceMethod
    raw_value: float  # Original calculated value before scaling
    num_tokens: int
    details: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "method": self.method.value,
            "raw_value": self.raw_value,
            "num_tokens": self.num_tokens,
            "details": self.details,
        }


class ConfidenceCalculator:
    """Calculate model confidence from token log probabilities.

    Based on: "Looking for a Needle in a Haystack: A Comprehensive Study of
    Hallucinations in Neural Machine Translation" (Guerreiro et al., 2023)

    Provides multiple methods for computing confidence scores from logprobs.
    """

    def __init__(self, default_method: ConfidenceMethod = ConfidenceMethod.GEOMETRIC):
        """Initialize calculator with default method.

        Args:
            default_method: Default calculation method to use.
        """
        self.default_method = default_method
        logger.info(f"ConfidenceCalculator initialized with method: {default_method}")

    def calculate(
        self,
        logprobs: list[float],
        method: ConfidenceMethod | None = None,
    ) -> ConfidenceResult:
        """Calculate confidence score from log probabilities.

        Args:
            logprobs: List of token log probabilities.
            method: Calculation method (uses default if not specified).

        Returns:
            ConfidenceResult with score and details.

        Raises:
            ValueError: If logprobs is empty or contains invalid values.
        """
        if not logprobs:
            raise ValueError("Cannot calculate confidence from empty logprobs")

        method = method or self.default_method
        logger.debug(
            f"Calculating confidence with method={method}, n_tokens={len(logprobs)}"
        )

        if method == ConfidenceMethod.AVERAGE:
            return self._average_confidence(logprobs)
        elif method == ConfidenceMethod.GEOMETRIC:
            return self._geometric_confidence(logprobs)
        elif method == ConfidenceMethod.MINIMUM:
            return self._minimum_confidence(logprobs)
        elif method == ConfidenceMethod.ENTROPY:
            return self._entropy_confidence(logprobs)
        elif method == ConfidenceMethod.VARIANCE:
            return self._variance_confidence(logprobs)
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    def calculate_all(self, logprobs: list[float]) -> dict[str, ConfidenceResult]:
        """Calculate confidence using all available methods.

        Args:
            logprobs: List of token log probabilities.

        Returns:
            Dictionary mapping method name to ConfidenceResult.
        """
        results = {}
        for method in ConfidenceMethod:
            try:
                results[method.value] = self.calculate(logprobs, method)
            except Exception as e:
                logger.warning(f"Failed to calculate {method}: {e}")
        return results

    def _average_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Average probability method.

        Converts each logprob to probability, takes average.
        Simple but can be skewed by a few high-confidence tokens.
        """
        probs = [math.exp(lp) for lp in logprobs]
        avg_prob = sum(probs) / len(probs)
        score = min(100.0, max(0.0, avg_prob * 100))

        return ConfidenceResult(
            score=round(score, 2),
            method=ConfidenceMethod.AVERAGE,
            raw_value=avg_prob,
            num_tokens=len(logprobs),
            details={
                "min_prob": min(probs),
                "max_prob": max(probs),
                "avg_prob": avg_prob,
            },
        )

    def _geometric_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Geometric mean (sequence probability) method.

        This is the standard approach: exp(mean(logprobs)).
        Represents the probability of the entire sequence.
        More sensitive to low-confidence tokens than average.
        """
        avg_logprob = sum(logprobs) / len(logprobs)
        seq_prob = math.exp(avg_logprob)
        score = min(100.0, max(0.0, seq_prob * 100))

        return ConfidenceResult(
            score=round(score, 2),
            method=ConfidenceMethod.GEOMETRIC,
            raw_value=seq_prob,
            num_tokens=len(logprobs),
            details={
                "avg_logprob": avg_logprob,
                "sequence_probability": seq_prob,
                "perplexity": math.exp(-avg_logprob),
            },
        )

    def _minimum_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Minimum probability method (pessimistic).

        Uses the least confident token as the score.
        Good for detecting hallucination-prone tokens.
        """
        min_logprob = min(logprobs)
        min_prob = math.exp(min_logprob)
        score = min(100.0, max(0.0, min_prob * 100))

        # Find which token(s) have minimum confidence
        min_indices = [i for i, lp in enumerate(logprobs) if lp == min_logprob]

        return ConfidenceResult(
            score=round(score, 2),
            method=ConfidenceMethod.MINIMUM,
            raw_value=min_prob,
            num_tokens=len(logprobs),
            details={
                "min_logprob": min_logprob,
                "min_prob": min_prob,
                "min_token_indices": min_indices,
                "num_low_confidence": len(min_indices),
            },
        )

    def _entropy_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Entropy-based uncertainty method.

        Higher entropy = more uncertainty = lower confidence.
        Normalized to 0-100 scale (inverted so high = confident).
        """
        probs = [math.exp(lp) for lp in logprobs]

        # Calculate entropy: -sum(p * log(p))
        # Handle numerical issues with small probabilities
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)

        # Normalize: max entropy for uniform distribution over vocab
        # Assuming a reasonable vocab size, max entropy ~ 10-12
        # We use a heuristic normalization
        max_entropy = 10.0  # Approximate max entropy for LLM tokens
        normalized_entropy = min(1.0, entropy / max_entropy)

        # Invert: low entropy = high confidence
        confidence = 1.0 - normalized_entropy
        score = confidence * 100

        return ConfidenceResult(
            score=round(score, 2),
            method=ConfidenceMethod.ENTROPY,
            raw_value=entropy,
            num_tokens=len(logprobs),
            details={
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "confidence": confidence,
            },
        )

    def _variance_confidence(self, logprobs: list[float]) -> ConfidenceResult:
        """Variance-based consistency method.

        Low variance in logprobs = consistent confidence = higher score.
        High variance suggests uncertainty about some tokens.
        """
        n = len(logprobs)
        mean_lp = sum(logprobs) / n
        variance = sum((lp - mean_lp) ** 2 for lp in logprobs) / n
        std_dev = math.sqrt(variance)

        # Convert to confidence: low variance = high confidence
        # Using exponential decay: exp(-variance)
        # Scale factor to make typical variances meaningful
        consistency = math.exp(-variance * 0.5)
        score = consistency * 100

        return ConfidenceResult(
            score=round(score, 2),
            method=ConfidenceMethod.VARIANCE,
            raw_value=variance,
            num_tokens=len(logprobs),
            details={
                "variance": variance,
                "std_dev": std_dev,
                "mean_logprob": mean_lp,
                "consistency": consistency,
            },
        )

    @staticmethod
    def interpret_score(score: float) -> str:
        """Interpret a confidence score.

        Args:
            score: Confidence score (0-100).

        Returns:
            Human-readable interpretation.
        """
        if score >= 80:
            return "High confidence - unlikely hallucination"
        elif score >= 60:
            return "Moderate confidence - some uncertainty"
        elif score >= 40:
            return "Low confidence - potential hallucination"
        elif score >= 20:
            return "Very low confidence - likely hallucination"
        else:
            return "Extremely low confidence - almost certainly hallucinated"

    @staticmethod
    def get_risk_level(score: float) -> str:
        """Get hallucination risk level from confidence score.

        Args:
            score: Confidence score (0-100).

        Returns:
            Risk level: "low", "medium", "high", or "critical".
        """
        if score >= 70:
            return "low"
        elif score >= 50:
            return "medium"
        elif score >= 30:
            return "high"
        else:
            return "critical"
