"""Privacy red teaming service.

Tests LLMs for privacy vulnerabilities including training data extraction,
membership inference, and prompt/system leakage.
"""

import logging

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from services.privacy_redteam.src.evaluator import (
    PrivacyEvaluationReport,
    PrivacyRedTeamEvaluator,
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


def privacy_redteam_service(
    model: Model | dict,
    test_types: list[str] | None = None,
    num_samples: int = 5,
    system_prompt: str | None = None,
    custom_probes: list[str] | None = None,
) -> dict:
    """Test model for privacy vulnerabilities.

    Performs active red-teaming tests to probe for:
    - Training data extraction: Tests if model leaks memorized data
    - Membership inference: Tests if model reveals training data presence
    - Prompt/system leakage: Tests if model reveals confidential instructions

    Args:
        model: Target LLM configuration.
        test_types: List of test types to run.
            Options: "training_extraction", "membership_inference", "prompt_leakage".
            If None, runs all tests.
        num_samples: Number of samples per test/category (1-20).
        system_prompt: Optional system prompt to test for leakage.
            If provided, tests will probe whether this prompt can be extracted.
        custom_probes: Optional custom extraction probes to test.

    Returns:
        Dictionary containing:
        - model: Model name
        - results: Results by test type
        - summary: Overall statistics and privacy grade
    """
    # Handle both Pydantic model and dict
    model_name = model.name if hasattr(model, "name") else model["name"]
    logger.info(f"Starting privacy red team test for model: {model_name}")

    # Create target model
    target_model = _create_target_model(model)

    # Create evaluator with all config options
    evaluator = PrivacyRedTeamEvaluator(
        test_types=test_types,
        num_samples=num_samples,
        system_prompt=system_prompt,
        custom_probes=custom_probes,
    )

    # Run evaluation
    report: PrivacyEvaluationReport = evaluator.evaluate(target_model)

    logger.info(
        f"Privacy red teaming complete: "
        f"Grade {report.summary.privacy_grade}, "
        f"Score {report.summary.overall_score:.2f}, "
        f"Leakages {report.summary.total_leakages}"
    )

    return report.to_dict()
