"""Factory for creating model wrappers from configuration."""

import logging
from typing import Any

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai

logger = logging.getLogger(__name__)


def create_target_model(model: dict[str, Any]):
    """Create the appropriate model wrapper based on model configuration.

    Args:
        model: Model configuration dict with 'name', 'description',
            and optionally 'base_url' for HuggingFace models.

    Returns:
        Model wrapper instance (APIModelOpenai or APIModelHuggingFace).

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
