"""Consistency and reliability testing service.

Tests LLM consistency through sycophancy detection, response stability,
self-consistency, and instruction following fidelity.
"""

import logging

from services.evaluation.benchmarks.consistency_reliability.src.evaluator import (
    ConsistencyReliabilityEvaluator,
    EvaluationReport,
)
from utils.model_factory import create_target_model
from utils.models import Model

logger = logging.getLogger(__name__)


def consistency_reliability_service(
    model: Model | dict,
    prompt: str,
    test_types: list[str] | None = None,
    num_samples: int = 5,
    sycophancy_topics: list[str] | None = None,
    instruction_constraints: list[str] | None = None,
) -> dict:
    """Test model consistency and reliability.

    Runs a suite of tests to evaluate how consistent and reliable
    a model's responses are under various conditions.

    Args:
        model: Target LLM configuration.
        prompt: Original prompt to test.
        test_types: List of test types to run.
            Options: "sycophancy", "stability", "self_consistency", "instruction_following".
            If None, runs all tests.
        num_samples: Number of samples/variations per test.
        sycophancy_topics: Custom topics for sycophancy testing.
        instruction_constraints: Custom constraints for instruction following test.

    Returns:
        Dictionary containing:
        - model: Model name
        - prompt: Original prompt
        - results: Results by test type
        - summary: Overall statistics and grade
    """
    # Handle both Pydantic model and dict
    model_name = model.name if hasattr(model, "name") else model["name"]
    logger.info(f"Starting consistency/reliability test for model: {model_name}")

    # Create target model
    target_model = create_target_model(model)

    # Create evaluator with all config options
    evaluator = ConsistencyReliabilityEvaluator(
        test_types=test_types,
        num_samples=num_samples,
        sycophancy_topics=sycophancy_topics,
        instruction_constraints=instruction_constraints,
    )

    # Run evaluation
    report: EvaluationReport = evaluator.evaluate(prompt, target_model)

    logger.info(
        f"Consistency/reliability testing complete: "
        f"Grade {report.summary['reliability_grade']}, "
        f"Score {report.summary['overall_score']:.2f}"
    )

    return report.to_dict()
