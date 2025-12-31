"""FActScore hallucination detection service."""

import logging
from typing import Any

from ..models.open_ai import OpenAIModel
from ..prompts.all_prompts import get_fact_checker_prompts

logger = logging.getLogger(__name__)


def detect(
    source: str,
    atomic_facts: list[dict[str, Any]],
    domain: str | None = None,
) -> dict[str, Any]:
    """
    Detect hallucinations by checking if atomic facts are supported by evidence.

    Uses the FActScore methodology (Min et al., 2023) to evaluate factual precision.

    Args:
        source: The evidence/source text to check facts against.
        atomic_facts: List of dicts with 'sentence' and 'facts' keys.
        domain: Optional domain restriction (e.g., "genomics", "medicine").
                If provided, facts outside this domain are marked as unsupported.

    Returns:
        Dictionary with sentences, per-sentence scores, and overall factscore.
    """
    # Get prompts with optional domain restriction
    prompts = get_fact_checker_prompts(domain=domain)

    model = OpenAIModel(
        name="gpt-3.5-turbo",
        prompts=prompts,
        fact_checker=True,
    )

    results: dict[str, dict[str, Any]] = {}

    for item in atomic_facts:
        sentence = item["sentence"]
        facts = item["facts"]

        # Filter out empty facts
        facts = [f.strip() for f in facts if f.strip()]

        if not facts:
            logger.warning(
                f"No valid facts extracted from sentence: {sentence[:50]}..."
            )
            continue

        if sentence not in results:
            results[sentence] = {"facts": facts, "supported": []}

        for fact in facts:
            evidence_and_fact = {
                "evidence": source,
                "fact": fact,
            }

            try:
                response = model.model_predict(data=evidence_and_fact)
                response_text = (
                    response[0].to_dict()["choices"][0]["message"]["content"].lower()
                )
                fact_supported = "true" in response_text
            except Exception as e:
                logger.error(f"Error checking fact '{fact[:50]}...': {e}")
                fact_supported = False

            results[sentence]["supported"].append(fact_supported)

    # Calculate per-sentence and overall scores
    result_json: dict[str, Any] = {
        "sentences": [],
        "scores": [],
        "details": [],
    }

    for sentence, data in results.items():
        facts = data["facts"]
        supported = data["supported"]

        if not facts:
            continue

        true_count = sum(1 for s in supported if s)
        total = len(facts)
        score = int(true_count / total * 100) if total > 0 else 0

        result_json["sentences"].append(sentence)
        result_json["scores"].append(score)
        result_json["details"].append(
            {
                "sentence": sentence,
                "facts": facts,
                "supported": supported,
                "supported_count": true_count,
                "total_facts": total,
            }
        )

    # Calculate overall FActScore
    if result_json["scores"]:
        factscore = int(sum(result_json["scores"]) / len(result_json["scores"]))
    else:
        factscore = 0
        logger.warning("No sentences with valid facts found")

    result_json["factscore"] = factscore
    result_json["domain"] = domain

    return result_json
