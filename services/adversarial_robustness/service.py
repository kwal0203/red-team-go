"""Adversarial robustness testing service.

Tests whether model safety guardrails can be bypassed through
various text perturbation techniques.
"""

import logging

from services.adversarial_robustness.src.evaluator import (
    AdversarialEvaluator,
    RobustnessReport,
)
from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from utils.models import Model

logger = logging.getLogger(__name__)


def _create_target_model(model: Model):
    """Create the appropriate model wrapper based on model configuration."""
    if "openai" in model["name"]:
        logger.info(f"Creating OpenAI model wrapper for {model['name']}")
        return APIModelOpenai(name=model["name"], description=model["description"])
    elif "huggingface" in model["name"]:
        logger.info(f"Creating HuggingFace model wrapper for {model['name']}")
        return APIModelHuggingFace(
            base_url=model["base_url"],
            name=model["name"],
            description=model["description"],
        )
    else:
        raise ValueError(
            f"Invalid model name '{model['name']}': must contain 'openai' or 'huggingface'"
        )


def adversarial_robustness_service(
    model: Model,
    prompt: str,
    perturbation_types: list[str] | None = None,
    num_variants: int = 5,
) -> dict:
    """Test model robustness against adversarial text perturbations.

    Generates perturbed variants of the input prompt using various
    techniques (character-level, word-level, semantic) and tests
    whether any variants can bypass the model's safety guardrails.

    Args:
        model: Target LLM configuration.
        prompt: Original prompt to test.
        perturbation_types: List of perturbation types to use.
            Options: "character", "word", "semantic".
            If None, uses all types.
        num_variants: Number of variants to generate per perturbation type.

    Returns:
        Dictionary containing:
        - original_prompt: The input prompt
        - original_blocked: Whether original would be blocked
        - variants: List of perturbation results
        - summary: Statistics including bypass rate
    """
    logger.info(f"Starting adversarial robustness test for model: {model['name']}")

    # Create target model
    target_model = _create_target_model(model)

    # Create evaluator
    evaluator = AdversarialEvaluator(
        perturbation_types=perturbation_types,
        num_variants=num_variants,
    )

    # Run evaluation
    report: RobustnessReport = evaluator.evaluate(prompt, target_model)

    logger.info(
        f"Adversarial testing complete: {report.summary['bypasses']} bypasses "
        f"out of {report.summary['total_variants']} variants"
    )

    return report.to_dict()
