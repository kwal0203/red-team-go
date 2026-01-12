"""Misinformation and factuality testing service.

Tests LLM factuality through knowledge cutoff detection, temporal reasoning,
confidence calibration, and citation verification.
"""

import logging

from services.detection.factuality.src.evaluator import (
    EvaluationReport,
    MisinformationFactualityEvaluator,
)
from utils.model_factory import create_target_model
from utils.models import Model

logger = logging.getLogger(__name__)


def misinformation_factuality_service(
    model: Model | dict,
    prompt: str,
    test_types: list[str] | None = None,
    num_samples: int = 5,
    knowledge_cutoff_date: str | None = None,
    temporal_questions: list[str] | None = None,
) -> dict:
    """Test model for misinformation and factuality issues.

    Runs a suite of tests to evaluate how factual and truthful
    a model's responses are.

    Args:
        model: Target LLM configuration.
        prompt: Base topic/context to test.
        test_types: List of test types to run.
            Options: "knowledge_cutoff", "temporal_reasoning",
                     "confidence_calibration", "citation_verification".
            If None, runs all tests.
        num_samples: Number of questions/samples per test.
        knowledge_cutoff_date: Expected cutoff date for verification.
        temporal_questions: Custom temporal reasoning questions.

    Returns:
        Dictionary containing:
        - model: Model name
        - prompt: Original prompt
        - results: Results by test type
        - summary: Overall statistics and grade
    """
    # Handle both Pydantic model and dict
    model_name = model.name if hasattr(model, "name") else model["name"]
    logger.info(f"Starting misinformation/factuality test for model: {model_name}")

    # Create target model
    target_model = create_target_model(model)

    # Create evaluator with all config options
    evaluator = MisinformationFactualityEvaluator(
        test_types=test_types,
        num_samples=num_samples,
        knowledge_cutoff_date=knowledge_cutoff_date,
        temporal_questions=temporal_questions,
    )

    # Run evaluation
    report: EvaluationReport = evaluator.evaluate(prompt, target_model)

    logger.info(
        f"Misinformation/factuality testing complete: "
        f"Grade {report.summary['factuality_grade']}, "
        f"Score {report.summary['overall_score']:.2f}"
    )

    return report.to_dict()
