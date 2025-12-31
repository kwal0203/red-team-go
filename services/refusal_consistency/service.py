"""Refusal consistency testing service.

Tests if LLMs consistently refuse harmful requests under adversarial conditions
including paraphrasing, user pressure, multi-turn escalation, and context switching.
"""

import logging

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from services.refusal_consistency.src.evaluator import (
    EvaluationReport,
    RefusalConsistencyEvaluator,
)
from utils.models import Model

logger = logging.getLogger(__name__)


def _create_target_model(model: Model | dict):
    """Create the appropriate model wrapper based on model configuration.

    Args:
        model: Model configuration (Pydantic model or dict).

    Returns:
        Model wrapper instance.

    Raises:
        ValueError: If model name is invalid.
    """
    # Handle both Pydantic model and dict
    if hasattr(model, "name"):
        model_name = model.name
        model_desc = model.description
        model_url = getattr(model, "base_url", None)
    else:
        model_name = model["name"]
        model_desc = model["description"]
        model_url = model.get("base_url")

    if "openai" in model_name:
        logger.info(f"Creating OpenAI model wrapper for {model_name}")
        return APIModelOpenai(name=model_name, description=model_desc)
    elif "huggingface" in model_name:
        logger.info(f"Creating HuggingFace model wrapper for {model_name}")
        return APIModelHuggingFace(
            base_url=model_url,
            name=model_name,
            description=model_desc,
        )
    else:
        raise ValueError(
            f"Invalid model name '{model_name}': must contain 'openai' or 'huggingface'"
        )


def refusal_consistency_service(
    model: Model | dict,
    test_types: list[str] | None = None,
    harm_categories: list[str] | None = None,
    num_samples: int = 3,
    custom_harmful_prompts: list[str] | None = None,
) -> dict:
    """Test model refusal consistency.

    Tests if a model consistently refuses harmful requests under various
    adversarial conditions including paraphrasing, user pressure,
    multi-turn escalation, and context switching.

    Args:
        model: Target LLM configuration.
        test_types: List of test types to run.
            Options: "paraphrase", "pressure", "multi_turn", "context_switching".
            If None, runs all tests.
        harm_categories: Harmful content categories to test.
            Options: "weapons", "illegal_activities", "self_harm",
                     "violence", "cbrn", "dual_use".
            If None, tests all categories.
        num_samples: Number of samples per test/category.
        custom_harmful_prompts: Optional custom harmful prompts to test.

    Returns:
        Dictionary containing:
        - model: Model name
        - harm_categories_tested: Categories that were tested
        - results: Results by test type
        - summary: Overall statistics and grade
    """
    # Handle both Pydantic model and dict
    model_name = model.name if hasattr(model, "name") else model["name"]
    logger.info(f"Starting refusal consistency test for model: {model_name}")

    # Create target model
    target_model = _create_target_model(model)

    # Create evaluator with all config options
    evaluator = RefusalConsistencyEvaluator(
        test_types=test_types,
        num_samples=num_samples,
        harm_categories=harm_categories,
        custom_harmful_prompts=custom_harmful_prompts,
    )

    # Run evaluation
    report: EvaluationReport = evaluator.evaluate(target_model)

    # Update harm_categories_tested with actual categories
    if harm_categories:
        report.harm_categories_tested = harm_categories
    else:
        # Default categories
        report.harm_categories_tested = [
            "weapons",
            "illegal_activities",
            "self_harm",
            "violence",
            "cbrn",
            "dual_use",
        ]

    logger.info(
        f"Refusal consistency testing complete: "
        f"Grade {report.summary.refusal_grade}, "
        f"Score {report.summary.overall_score:.2f}"
    )

    return report.to_dict()
