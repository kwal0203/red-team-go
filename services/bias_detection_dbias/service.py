"""Bias detection service for evaluating LLM responses."""

import logging
from typing import Any

from services.bias_detection_dbias.src.bias_detection import (
    detect_bias,
    detect_bias_single,
)
from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from utils.models import BiasScore, Model, ResultRealtimeBias

logger = logging.getLogger(__name__)


def _create_target_model(model: Model):
    """Create the appropriate model wrapper based on model configuration."""
    if "openai" in model["name"]:
        logger.info("Creating OpenAI model wrapper")
        return APIModelOpenai(name=model["name"], description=model["description"])
    elif "huggingface" in model["name"]:
        logger.info("Creating HuggingFace model wrapper")
        return APIModelHuggingFace(
            base_url=model["base_url"],
            name=model["name"],
            description=model["description"],
        )
    else:
        raise ValueError(
            f"Invalid model name '{model['name']}': must contain 'openai' or 'huggingface'"
        )


def dbias_service(
    model: Model,
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
    """
    try:
        target_model = _create_target_model(model)
    except ValueError as e:
        logger.error(str(e))
        return {"bias_evaluation": str(e)}

    logger.info(f"Evaluating {len(user_prompts)} prompts for bias")
    inputs = [{"dataset": "user_provided", "prompt": p} for p in user_prompts]

    results = detect_bias(model=target_model, inputs=inputs)
    return {"bias_evaluation": results}


def bias_detection_realtime_service(
    model: Model,
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
    """
    # Create target model
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
