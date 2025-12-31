from typing import List, Tuple
from src.context import SemanticEntropyContext
from src.utility.response import response_generator


def make_prompt(context: str, question: str) -> str:
    """Create a prompt for QA-style semantic entropy evaluation."""
    prompt = ""
    prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"
    return prompt


def get_generations(
    context: SemanticEntropyContext,
    prompt: str,
    num_generations: int = 10,
    temperature: float = 0.5,
) -> List[Tuple[str, List]]:
    """
    Generate multiple responses for semantic entropy calculation.

    We sample multiple high temperature answers which will be used to estimate
    the semantic entropy - responses that are semantically equivalent indicate
    higher model confidence.

    Args:
        context: SemanticEntropyContext containing model client and configuration.
        prompt: The input prompt to generate responses for.
        num_generations: Number of responses to generate.
        temperature: Sampling temperature (higher = more diverse).

    Returns:
        List of tuples containing (response_text, log_likelihoods).
    """
    responses = []
    for _ in range(num_generations):
        response = response_generator(context=context, prompt=prompt)
        response_dict = response.to_dict()
        predicted_answer = response_dict["choices"][0]["message"]["content"]
        log_likelihoods = response_dict["choices"][0]["logprobs"]["content"]
        responses.append((predicted_answer, log_likelihoods))

    return responses
