from typing import Any, Dict, Optional
import openai

from src.models.entailment_model import EntailmentDeberta
from src.services.semantic_ids import get_semantic_ids
from src.utility.misc import get_generations
from src.context import SemanticEntropyContext
from utils.config import get_openai_key


def semantic_entropy_service(
    prompt: str,
    model_name: str = "gpt-3.5-turbo",
    num_generations: int = 5,
    entailment_model_type: str = "deberta",
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Semantic entropy hallucination detection service.

    Based on "Semantic Uncertainty: Linguistic Invariances for Uncertainty
    Estimation in Natural Language Generation" (Kuhn et al., 2023).

    The idea is that if a model is uncertain about an answer, it will generate
    semantically different responses when sampled multiple times. We use an
    entailment model to group responses by semantic equivalence and compute
    a score based on the number of distinct semantic clusters.

    Args:
        prompt: The input prompt to evaluate.
        model_name: The LLM model to use (e.g., "gpt-3.5-turbo", "llama3-instruct").
        num_generations: Number of responses to generate for entropy calculation.
        entailment_model_type: Type of entailment model ("deberta").
        base_url: Optional base URL for custom API endpoints (e.g., TGI).

    Returns:
        Dict containing:
            - semantic_entropy: Score from 0-100 (higher = more consistent/confident)
            - num_clusters: Number of semantically distinct response clusters
            - num_generations: Total number of generations used
    """
    # Initialize the LLM client
    if model_name == "llama3-instruct" and base_url:
        # HuggingFace TGI endpoint
        model_client = openai.OpenAI(base_url=base_url, api_key="dummy")
    else:
        # OpenAI API
        openai.api_key = get_openai_key()
        model_client = openai

    # Create context for generation
    context = SemanticEntropyContext(
        model_client=model_client,
        model_name=model_name,
        one_shot=False,
        prompts=None,
        messages=[],
    )

    # Load entailment model for semantic similarity
    if entailment_model_type == "deberta":
        entailment_model = EntailmentDeberta()
    else:
        raise ValueError(f"Unknown entailment model type: {entailment_model_type}")

    # Generate multiple responses
    generations = get_generations(
        context=context,
        prompt=prompt,
        num_generations=num_generations,
        temperature=0.9,
    )

    # Extract response texts
    responses = [response[0] for response in generations]

    # Compute semantic IDs (cluster responses by semantic equivalence)
    semantic_ids = get_semantic_ids(strings_list=responses, model=entailment_model)

    # Calculate semantic entropy score
    # More unique clusters = higher uncertainty = lower score
    num_clusters = len(set(semantic_ids))
    semantic_id_score = -(num_clusters / len(generations)) + 1
    semantic_id_score = int(semantic_id_score * 100)

    return {
        "semantic_entropy": semantic_id_score,
        "num_clusters": num_clusters,
        "num_generations": len(generations),
    }
