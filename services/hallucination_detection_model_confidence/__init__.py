"""Model confidence hallucination detection service.

Based on: "Looking for a Needle in a Haystack: A Comprehensive Study of
Hallucinations in Neural Machine Translation" (Guerreiro et al., 2023)
"""

from .service import (
    ModelConfidenceResult,
    evaluate_confidence,
    evaluate_text_confidence,
    service,
)
from .src.confidence_calculator import ConfidenceCalculator, ConfidenceMethod
from .src.logprobs_client import LogprobsClient

__all__ = [
    "service",
    "evaluate_confidence",
    "evaluate_text_confidence",
    "ModelConfidenceResult",
    "ConfidenceCalculator",
    "ConfidenceMethod",
    "LogprobsClient",
]
