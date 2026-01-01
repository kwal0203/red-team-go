"""Bias detection service for evaluating LLM responses."""

import logging
from typing import Any

from services.bias_detection_dbias.src.bias_detection import (
    detect_bias,
    detect_bias_single,
)
from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from utils.models import BiasScore, ResultRealtimeBias

logger = logging.getLogger(__name__)


def _create_target_model(model: dict[str, Any]):
    """Create the appropriate model wrapper based on model configuration.

    Raises:
        ValueError: If model name/description missing or model type invalid.
    """
    model_name = model.get("name")
    model_desc = model.get("description")

    if not model_name or not model_desc:
        raise ValueError("Model name and description are required")

    if "openai" in model_name:
        logger.info(f"Creating OpenAI model wrapper for {model_name}")
        return APIModelOpenai(name=model_name, description=model_desc)
    elif "huggingface" in model_name:
        base_url = model.get("base_url")
        if not base_url:
            raise ValueError(
                f"base_url is required for HuggingFace model '{model_name}'"
            )
        logger.info(f"Creating HuggingFace model wrapper for {model_name}")
        return APIModelHuggingFace(
            base_url=base_url,
            name=model_name,
            description=model_desc,
        )
    else:
        raise ValueError(
            f"Invalid model name '{model_name}': must contain 'openai' or 'huggingface'"
        )


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
    target_model = _create_target_model(model)

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
    target_model = _create_target_model(model)

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
