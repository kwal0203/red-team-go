"""Refusal consistency testing service.

Tests if LLMs consistently refuse harmful requests under adversarial conditions
including paraphrasing, user pressure, multi-turn escalation, and context switching.
"""

import logging

from services.evaluation.benchmarks.refusal_consistency.src.evaluator import (
    EvaluationReport,
    RefusalConsistencyEvaluator,
)
from utils.model_factory import create_target_model
from utils.models import Model

logger = logging.getLogger(__name__)


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
    target_model = create_target_model(model)

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
