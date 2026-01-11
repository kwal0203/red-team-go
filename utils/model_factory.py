"""Factory for creating model wrappers from configuration."""

import logging
from typing import Any

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from services.model_wrappers.model_openrouter import APIModelOpenRouter
from utils.config import (
    get_default_model_provider,
    get_openai_model_name,
    get_openrouter_model_name,
)

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

    provider: str | None = None
    if "openai" in model_name:
        provider = "openai"
    elif "openrouter" in model_name:
        provider = "openrouter"
    elif "huggingface" in model_name:
        provider = "huggingface"
    elif model_name == "default":
        provider = get_default_model_provider()

    if provider == "openai":
        logger.info(f"Creating OpenAI model wrapper for {model_name}")
        model_id = model.get("model_name") or get_openai_model_name()
        return APIModelOpenai(
            name=model_name,
            description=model_desc,
            model_name=model_id,
        )
    elif provider == "openrouter":
        logger.info(f"Creating OpenRouter model wrapper for {model_name}")
        model_id = model.get("model_name") or get_openrouter_model_name()
        return APIModelOpenRouter(
            name=model_name,
            description=model_desc,
            model_name=model_id,
        )
    elif provider == "huggingface":
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
            f"Invalid model name '{model_name}': must contain 'openai', 'openrouter', or 'huggingface' (or use name 'default' with DEFAULT_MODEL_PROVIDER)"
        )
