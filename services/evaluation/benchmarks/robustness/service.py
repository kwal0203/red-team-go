"""Adversarial robustness testing service.

Tests whether model safety guardrails can be bypassed through
various text perturbation techniques.
"""

import logging

from services.evaluation.benchmarks.robustness.src.evaluator import (
    AdversarialEvaluator,
    RobustnessReport,
)
from utils.model_factory import create_target_model
from utils.models import Model

logger = logging.getLogger(__name__)


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
    target_model = create_target_model(model)

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
