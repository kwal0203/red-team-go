"""Factory for creating model wrappers from configuration."""

import logging
from collections.abc import Mapping
from typing import Any

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from services.model_wrappers.model_openrouter import APIModelOpenRouter
from utils.config import (
    get_default_model_provider,
    get_openai_model_name,
    get_openrouter_model_name,
)
from utils.models import Model as ModelConfig

logger = logging.getLogger(__name__)


def _normalize_model_config(model: ModelConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic Model or plain mapping into a dict."""
    if isinstance(model, ModelConfig):
        return model.model_dump()
    if isinstance(model, Mapping):
        return dict(model)
    raise ValueError("Model configuration must be a mapping or utils.models.Model")


def create_target_model(model: ModelConfig | Mapping[str, Any]):
    """Create the appropriate model wrapper based on model configuration.

    Args:
        model: Model configuration with at least 'name' and 'description'.
            Accepts both the Pydantic Model object and plain dicts.

    Returns:
        Model wrapper instance for the requested provider.

    Raises:
        ValueError: If required fields are missing or provider is invalid.
    """
    model_dict = _normalize_model_config(model)
    model_name = model_dict.get("name")
    model_desc = model_dict.get("description")
    base_url = model_dict.get("base_url")
    model_id = model_dict.get("model_name")
    provider_override = model_dict.get("provider", None) or model_dict.get(
        "backend", None
    )

    if not model_name or not model_desc:
        raise ValueError("Model name and description are required")

    name_lower = model_name.lower()
    provider: str | None = None
    if provider_override:
        provider = str(provider_override).lower()
    elif "openai" in name_lower:
        provider = "openai"
    elif "openrouter" in name_lower:
        provider = "openrouter"
    elif "huggingface" in name_lower or base_url:
        provider = "huggingface"
    elif model_name == "default":
        provider = get_default_model_provider()

    if provider == "openai":
        logger.info(f"Creating OpenAI model wrapper for {model_name}")
        return APIModelOpenai(
            name=model_name,
            description=model_desc,
            model_name=model_id or get_openai_model_name(),
        )
    if provider == "openrouter":
        logger.info(f"Creating OpenRouter model wrapper for {model_name}")
        return APIModelOpenRouter(
            name=model_name,
            description=model_desc,
            model_name=model_id or get_openrouter_model_name(),
        )
    if provider == "huggingface":
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

    valid = (
        "openai",
        "openrouter",
        "huggingface",
    )
    raise ValueError(
        f"Invalid provider for model '{model_name}'. "
        f"Supported providers: {valid}. "
        "Name should include provider hint or set DEFAULT_MODEL_PROVIDER."
    )
