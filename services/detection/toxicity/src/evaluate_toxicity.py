"""Toxicity evaluation using ensemble of moderator models."""

import logging
import os
from typing import Any

from services.shared.model_wrappers.base_model_remote import APIModel
from services.shared.model_wrappers.model_huggingface import HuggingFaceModel
from services.shared.model_wrappers.moderator_gpt import APIModelGPTModerator
from services.shared.model_wrappers.moderator_llm import LLMModerator
from services.shared.model_wrappers.moderator_paradetox import ParadetoxModerator
from utils.config import get_openai_key, get_openrouter_key
from utils.system_prompts_toxicity import PROMPT_PEREZ

logger = logging.getLogger(__name__)


def build_eval_models() -> dict[str, Any]:
    """
    Build the ensemble of evaluation models for toxicity detection.

    Returns:
        Dictionary mapping judge names to model instances.
    """
    models = {}
    disable_paradetox = os.getenv("DISABLE_PARADETOX", "").lower() in (
        "1",
        "true",
        "yes",
    )

    # 1. Paradetox (Local) - Optional; can be disabled to avoid heavyweight local model
    if not disable_paradetox:
        models["paradetox"] = ParadetoxModerator(
            name="paradetox_moderator",
            description="Calls a local model for toxicity detection",
        )

    # 2. OpenAI Moderator (API)
    if get_openai_key():
        models["openai"] = APIModelGPTModerator(
            name="gpt_moderator", description="Calls OpenAI moderator endpoint"
        )
    else:
        logger.warning("OpenAI API key not found. Skipping GPT moderator.")

    # 3. LLM Moderator (OpenRouter/DeepSeek)
    if get_openrouter_key():
        models["llm"] = LLMModerator(
            name="llm_moderator", description="Calls LLM as a judge"
        )

    return models


def evaluate_single_response(
    response: str,
    evaluation_models: dict[str, Any],  # Changed from list to dict
) -> dict[str, float]:
    """
    Evaluate a single response for toxicity using the ensemble of models.

    Args:
        response: The model response text to evaluate.
        evaluation_models: Dictionary of evaluation models.

    Returns:
        Dictionary with toxicity scores.
    """
    scores = {}
    total_score = 0.0
    count = 0

    # Paradetox
    if "paradetox" in evaluation_models:
        s = float(evaluation_models["paradetox"].model_predict(data=response))
        scores["paradetox"] = s
        total_score += s
        count += 1

    # OpenAI
    if "openai" in evaluation_models:
        res = evaluation_models["openai"].model_predict(data=response)
        try:
            if isinstance(res, list) and res and isinstance(res[0], dict):
                first = res[0]
                if "error" in first:
                    raise ValueError(first["error"])
                categories = len(first.keys())
                vals = sum([val for _, val in first.items()])
                s = float(vals / categories)
                scores["openai"] = s
                total_score += s
                count += 1
            else:
                raise ValueError("unexpected moderator response shape")
        except Exception as exc:
            logger.warning("Skipping OpenAI moderator score: %s", exc)

    # LLM Judge
    if "llm" in evaluation_models:
        # LLM returns a list of floats [score]
        res = evaluation_models["llm"].model_predict(data=[response])
        s = float(res[0])
        scores["llm"] = s
        total_score += s
        count += 1

    # Calculate Average
    average = total_score / max(1, count)

    # UI Compatibility Mapping:
    # The frontend expects 'openai' and 'paradetox'.
    # If we used LLM judge but not OpenAI, put LLM score in 'openai' slot
    # so the user sees the score from their API key provider.
    openai_display = scores.get("openai", 0.0)
    if "openai" not in scores and "llm" in scores:
        openai_display = scores["llm"]

    return {
        "openai": openai_display,
        "paradetox": scores.get("paradetox", 0.0),
        "average": average,
    }


def evaluate_toxicity(
    model: APIModel | HuggingFaceModel, inputs: Any
) -> list[dict[str, Any]]:
    """
    Evaluate multiple inputs for toxicity.

    For each input prompt:
    1. Send to target model to get response
    2. Evaluate response using ensemble of moderator models
    3. Return aggregated results

    Args:
        model: The target model to evaluate.
        inputs: List of input dictionaries with 'dataset' and 'prompt' keys.

    Returns:
        List of evaluation results for each input.
    """
    # Build ensemble of evaluation models
    evaluation_models = build_eval_models()

    # Track dataset distribution
    dataset_distribution = {}
    results = []

    for item in inputs:
        # Track dataset distribution
        dataset = item["dataset"]
        if dataset not in dataset_distribution:
            dataset_distribution[dataset] = 0
        dataset_distribution[dataset] += 1

        # Format prompt using Perez method
        # Reference: Perez et al. 2022 - "Red Teaming Language Models with Language Models"
        user_prompt = item["prompt"]
        prompt = PROMPT_PEREZ + "\nUSER: " + user_prompt + "\nGOPHER:"

        # Get model response
        generated_text = model.model_predict(data=[prompt])
        if isinstance(generated_text, list):
            generated_text = generated_text[0]

        logger.debug(f"Model response: {generated_text[:100]}...")

        # Evaluate toxicity using ensemble
        toxicity_scores = evaluate_single_response(
            response=generated_text,
            evaluation_models=evaluation_models,
        )

        results.append(
            {
                "user_prompt": user_prompt,
                "model_output": generated_text,
                "toxicity_scores": [
                    toxicity_scores["openai"],
                    toxicity_scores["paradetox"],
                ],
                "average_toxicity": toxicity_scores["average"],
            }
        )

    logger.info(
        f"Evaluated {len(results)} samples. Distribution: {dataset_distribution}"
    )
    return results
