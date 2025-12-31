"""Stereotype benchmark evaluation service.

Evaluates models on established bias benchmarks including
StereoSet, CrowS-Pairs, and BBQ.
"""

import logging

from services.model_wrappers.model_huggingface_remote import APIModelHuggingFace
from services.model_wrappers.model_openai import APIModelOpenai
from services.stereotype_benchmarks.src.evaluator import (
    BenchmarkReport,
    StereotypeBenchmarkEvaluator,
)
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


def stereotype_benchmark_service(
    model: Model,
    benchmark: str,
    num_samples: int | None = None,
    bias_types: list[str] | None = None,
    include_samples: bool = False,
) -> dict:
    """Evaluate a model on a stereotype benchmark.

    Runs the specified benchmark and computes bias metrics.

    Args:
        model: Target LLM configuration.
        benchmark: Benchmark name ("stereoset", "crows_pairs", "bbq").
        num_samples: Number of samples to evaluate. If None, uses all available.
        bias_types: Filter to specific bias types (e.g., ["gender", "race"]).
        include_samples: Whether to include per-sample results in output.

    Returns:
        Dictionary containing:
        - benchmark: Benchmark name
        - model: Model name
        - num_samples: Samples evaluated
        - metrics: Overall metrics (stereotype_score, etc.)
        - by_bias_type: Metrics broken down by bias type
        - sample_results: Per-sample results (if include_samples=True)

    Raises:
        ValueError: If benchmark name is not recognized.
    """
    logger.info(f"Starting {benchmark} evaluation for model: {model['name']}")

    # Create target model
    target_model = _create_target_model(model)

    # Create evaluator
    try:
        evaluator = StereotypeBenchmarkEvaluator(benchmark=benchmark)
    except ValueError as e:
        logger.error(f"Invalid benchmark: {e}")
        raise

    # Run evaluation
    report: BenchmarkReport = evaluator.evaluate(
        model=target_model,
        num_samples=num_samples,
        bias_types=bias_types,
        include_samples=include_samples,
    )

    logger.info(
        f"{benchmark} evaluation complete: "
        f"stereotype_score={report.metrics.get('stereotype_score', 'N/A')}"
    )

    return report.to_dict()
