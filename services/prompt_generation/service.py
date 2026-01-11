"""Adversarial prompt generation service.

Generates adversarial prompts for red-teaming LLMs using various methods.
"""

import logging
from typing import Any

from services.prompt_generation.src.evaluator import (
    GenerationReport,
    PromptEvaluator,
)
from services.prompt_generation.src.generators import (
    AARTPromptGenerator,
    AdvPrompterPromptGenerator,
    AutoDANPromptGenerator,
    BlackBoxPAIRPromptGenerator,
    ColdAttackPromptGenerator,
    CRTPromptGenerator,
    DatasetPromptGenerator,
    DSNPromptGenerator,
    GeneticPromptGenerator,
    GPTFuzzerPromptGenerator,
    JailbreakHubGenerator,
    LLMPromptGenerator,
    ManyShotPromptGenerator,
    PAIRPromptGenerator,
    SAPPromptGenerator,
    STPPromptGenerator,
)
from utils.model_factory import create_target_model

logger = logging.getLogger(__name__)

# Registry of available generators
GENERATOR_REGISTRY = {
    "llm": LLMPromptGenerator,
    "genetic": GeneticPromptGenerator,
    "pair": PAIRPromptGenerator,
    # Experimental methods (prototypes under /experimental)
    "sap": SAPPromptGenerator,
    "aart": AARTPromptGenerator,
    "stp": STPPromptGenerator,
    "dsn": DSNPromptGenerator,
    "manyshot": ManyShotPromptGenerator,
    "advprompter": AdvPrompterPromptGenerator,
    "autodan": AutoDANPromptGenerator,
    "cold": ColdAttackPromptGenerator,
    "crt": CRTPromptGenerator,
    "gptfuzzer": GPTFuzzerPromptGenerator,
    "jailbreakhub": JailbreakHubGenerator,
    "blackbox_pair": BlackBoxPAIRPromptGenerator,
    "datasets": DatasetPromptGenerator,
}


def prompt_generation_service(
    model: dict[str, Any],
    target_category: str,
    generator: str = "llm",
    num_prompts: int = 10,
    seed_prompt: str | None = None,
    evaluate: bool = True,
) -> dict:
    """Generate adversarial prompts for red-teaming.

    Uses the specified generator to create adversarial prompts
    and optionally evaluates them against the target model.

    Generators:
        - Core: "llm", "genetic", "pair"
        - Experimental prototypes under /experimental:
          "sap", "aart", "stp", "dsn", "manyshot", "advprompter",
          "autodan", "cold", "crt", "gptfuzzer", "jailbreakhub",
          "blackbox_pair", "datasets"

    Args:
        model: Target LLM configuration.
        target_category: Category of prompts to generate
            (e.g., "jailbreak", "harmful", "bias", "toxicity").
        generator: Generator method to use.
        num_prompts: Number of prompts to generate.
        seed_prompt: Optional seed prompt for generation.
        evaluate: Whether to evaluate prompts against the target model.

    Returns:
        Dictionary containing:
        - generator: Generator method used
        - target_category: Category targeted
        - prompts: List of generated prompts with evaluations
        - summary: Statistics including bypass rate

    Raises:
        ValueError: If generator name is not recognized.
    """
    logger.info(
        f"Starting prompt generation: {num_prompts} prompts, "
        f"category={target_category}, generator={generator}"
    )

    # Validate generator
    if generator not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown generator: {generator}. "
            f"Available: {list(GENERATOR_REGISTRY.keys())}"
        )

    # Create target model
    target_model = create_target_model(model)

    # Create generator
    generator_class = GENERATOR_REGISTRY[generator]
    prompt_generator = generator_class()

    # Set model for LLM generator
    if hasattr(prompt_generator, "model"):
        prompt_generator.model = target_model

    # Generate prompts
    try:
        prompts = prompt_generator.generate(
            category=target_category,
            num_prompts=num_prompts,
            seed_prompt=seed_prompt,
        )
    except NotImplementedError as e:
        logger.warning(f"Generator not implemented: {e}")
        return {
            "generator": generator,
            "target_category": target_category,
            "prompts": [],
            "summary": {
                "total_generated": 0,
                "error": str(e),
            },
        }

    logger.info(f"Generated {len(prompts)} prompts")

    # Evaluate prompts if requested
    evaluations = None
    if evaluate and prompts:
        logger.info("Evaluating generated prompts")
        evaluator = PromptEvaluator()
        evaluations = evaluator.evaluate_prompts(prompts, target_model)

    # Create report
    evaluator = PromptEvaluator()
    report: GenerationReport = evaluator.create_report(
        generator_name=generator,
        category=target_category,
        prompts=prompts,
        evaluations=evaluations,
    )

    logger.info(
        f"Prompt generation complete: {report.summary['total_generated']} generated, "
        f"{report.summary['successful_bypasses']} bypasses"
    )

    return report.to_dict()
