"""Model Confidence hallucination detection service.

Based on: "Looking for a Needle in a Haystack: A Comprehensive Study of
Hallucinations in Neural Machine Translation" (Guerreiro et al., 2023)

This service evaluates the likelihood of hallucination in model outputs by
analyzing token-level log probabilities. Lower confidence suggests higher
risk of hallucination.

Confidence Methods:
- geometric: Sequence probability (default, most robust)
- average: Mean token probability
- minimum: Worst-case token confidence (pessimistic)
- entropy: Information-theoretic uncertainty
- variance: Consistency of confidence across tokens
"""

import logging
from dataclasses import dataclass
from typing import Any

from .src.confidence_calculator import ConfidenceCalculator, ConfidenceMethod
from .src.logprobs_client import LogprobsClient, LogprobsResponse

logger = logging.getLogger(__name__)


@dataclass
class ModelConfidenceResult:
    """Result from model confidence evaluation."""

    confidence_score: float
    risk_level: str
    interpretation: str
    method: str
    generated_text: str
    num_tokens: int
    details: dict
    all_methods: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "confidence_score": self.confidence_score,
            "risk_level": self.risk_level,
            "interpretation": self.interpretation,
            "method": self.method,
            "generated_text": self.generated_text,
            "num_tokens": self.num_tokens,
            "details": self.details,
        }
        if self.all_methods:
            result["all_methods"] = self.all_methods
        return result


def evaluate_confidence(
    prompt: str,
    model_name: str = "gpt-3.5-turbo",
    method: str = "geometric",
    system_prompt: str | None = None,
    max_tokens: int = 256,
    include_all_methods: bool = False,
) -> ModelConfidenceResult:
    """Evaluate model confidence for a given prompt.

    Generates a response and analyzes token-level log probabilities
    to estimate hallucination risk.

    Args:
        prompt: The prompt to send to the model.
        model_name: OpenAI model to use (default: gpt-3.5-turbo).
        method: Confidence calculation method (default: geometric).
            Options: average, geometric, minimum, entropy, variance
        system_prompt: Optional system prompt for context.
        max_tokens: Maximum tokens to generate.
        include_all_methods: If True, calculate all methods for comparison.

    Returns:
        ModelConfidenceResult with confidence score and risk assessment.

    Raises:
        ValueError: If prompt is empty or method is invalid.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Validate method
    try:
        confidence_method = ConfidenceMethod(method)
    except ValueError as e:
        valid_methods = [m.value for m in ConfidenceMethod]
        raise ValueError(
            f"Invalid method '{method}'. Valid options: {valid_methods}"
        ) from e

    logger.info(
        f"Evaluating confidence: model={model_name}, method={method}, "
        f"prompt_len={len(prompt)}"
    )

    # Get model response with logprobs
    client = LogprobsClient(model_name=model_name)
    response: LogprobsResponse = client.generate_with_logprobs(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic for consistency
    )

    # Calculate confidence
    calculator = ConfidenceCalculator(default_method=confidence_method)
    result = calculator.calculate(response.logprobs, confidence_method)

    # Optionally calculate all methods
    all_methods_dict = None
    if include_all_methods:
        all_results = calculator.calculate_all(response.logprobs)
        all_methods_dict = {name: res.to_dict() for name, res in all_results.items()}

    confidence_result = ModelConfidenceResult(
        confidence_score=result.score,
        risk_level=calculator.get_risk_level(result.score),
        interpretation=calculator.interpret_score(result.score),
        method=method,
        generated_text=response.text,
        num_tokens=response.total_tokens,
        details=result.details,
        all_methods=all_methods_dict,
    )

    logger.info(
        f"Confidence evaluation complete: score={result.score}, "
        f"risk={confidence_result.risk_level}"
    )

    return confidence_result


def evaluate_text_confidence(
    text: str,
    logprobs: list[dict[str, float]],
    method: str = "geometric",
    include_all_methods: bool = False,
) -> ModelConfidenceResult:
    """Evaluate confidence for pre-computed logprobs.

    Use this when you already have logprobs from a model response.
    This is the legacy interface matching the original service.

    Args:
        text: The generated text (for reference).
        logprobs: List of dicts with 'logprob' key containing log probabilities.
        method: Confidence calculation method.
        include_all_methods: If True, calculate all methods.

    Returns:
        ModelConfidenceResult with confidence score.

    Raises:
        ValueError: If logprobs is empty or malformed.
    """
    if not logprobs:
        raise ValueError("Logprobs list cannot be empty")

    # Extract logprob values
    try:
        logprob_values = [lp["logprob"] for lp in logprobs]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Invalid logprobs format. Expected list of dicts with 'logprob' key: {e}"
        ) from e

    # Validate method
    try:
        confidence_method = ConfidenceMethod(method)
    except ValueError as e:
        valid_methods = [m.value for m in ConfidenceMethod]
        raise ValueError(
            f"Invalid method '{method}'. Valid options: {valid_methods}"
        ) from e

    logger.info(
        f"Evaluating text confidence: n_tokens={len(logprobs)}, method={method}"
    )

    calculator = ConfidenceCalculator(default_method=confidence_method)
    result = calculator.calculate(logprob_values, confidence_method)

    all_methods_dict = None
    if include_all_methods:
        all_results = calculator.calculate_all(logprob_values)
        all_methods_dict = {name: res.to_dict() for name, res in all_results.items()}

    return ModelConfidenceResult(
        confidence_score=result.score,
        risk_level=calculator.get_risk_level(result.score),
        interpretation=calculator.interpret_score(result.score),
        method=method,
        generated_text=text,
        num_tokens=len(logprobs),
        details=result.details,
        all_methods=all_methods_dict,
    )


def service(args: Any) -> dict:
    """Legacy service interface for backwards compatibility.

    Supports both new prompt-based evaluation and legacy logprobs-based.

    Args:
        args: Object with either:
            - prompt (str): Prompt to evaluate
            - model (str, optional): Model name
            - method (str, optional): Calculation method
            OR (legacy):
            - logprobs: Pre-computed logprobs

    Returns:
        Dictionary with confidence evaluation results.
    """
    # Check for new prompt-based interface
    if hasattr(args, "prompt") and args.prompt:
        model_name = getattr(args, "model_name", "gpt-3.5-turbo")
        method = getattr(args, "method", "geometric")
        system_prompt = getattr(args, "system_prompt", None)
        max_tokens = getattr(args, "max_tokens", 256)
        include_all = getattr(args, "include_all_methods", False)

        result = evaluate_confidence(
            prompt=args.prompt,
            model_name=model_name,
            method=method,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            include_all_methods=include_all,
        )
        return result.to_dict()

    # Legacy logprobs interface
    if hasattr(args, "logprobs"):
        method = getattr(args, "method", "geometric")
        text = getattr(args, "text", "")
        include_all = getattr(args, "include_all_methods", False)

        result = evaluate_text_confidence(
            text=text,
            logprobs=args.logprobs,
            method=method,
            include_all_methods=include_all,
        )
        return result.to_dict()

    raise ValueError(
        "Invalid arguments. Provide either 'prompt' for new evaluation "
        "or 'logprobs' for pre-computed evaluation."
    )
