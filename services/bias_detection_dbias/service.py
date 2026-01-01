"""Bias detection service for evaluating LLM responses."""

import logging
from typing import Any

from services.bias_detection_dbias.src.bias_detection import (
    detect_bias,
    detect_bias_single,
)
from utils.model_factory import create_target_model
from utils.models import BiasScore, ResultRealtimeBias

logger = logging.getLogger(__name__)


def dbias_service(
    model: dict[str, Any],
    user_prompts: list[str],
) -> dict[str, Any]:
    """
    Batch bias detection service using DBias methodology.

    Based on: Raza et al. "Dbias: detecting biases and ensuring fairness in news articles"
    Int J Data Sci Anal 17, 39-59 (2024).

    Args:
        model: Target LLM configuration.
        user_prompts: List of prompts to evaluate.

    Returns:
        Dictionary containing bias evaluation results.

    Raises:
        ValueError: If model configuration is invalid.
    """
    target_model = create_target_model(model)

    logger.info(f"Evaluating {len(user_prompts)} prompts for bias")
    inputs = [{"dataset": "user_provided", "prompt": p} for p in user_prompts]

    results = detect_bias(model=target_model, inputs=inputs)
    return {"bias_evaluation": results}


def bias_detection_realtime_service(
    model: dict[str, Any],
    prompt: str,
) -> ResultRealtimeBias:
    """
    Realtime bias detection service for a single prompt.

    Sends the prompt to the target model, evaluates the response for bias
    using self-evaluation, and returns both the response and bias assessment.

    Args:
        model: Target LLM configuration.
        prompt: The prompt to send to the model and evaluate.

    Returns:
        ResultRealtimeBias containing the prompt, response, and bias assessment.

    Raises:
        ValueError: If model configuration is invalid.
    """
    # Create target model (let ValueError propagate for endpoint to handle)
    target_model = create_target_model(model)

    # Get model response and evaluate for bias
    logger.info(f"Sending prompt to target model: {model['name']}")
    result = detect_bias_single(model=target_model, prompt=prompt)

    return ResultRealtimeBias(
        prompt=prompt,
        model_response=result["model_response"],
        bias=BiasScore(
            bias_detected=result["bias_detected"],
            explanation=result["explanation"],
        ),
    )
