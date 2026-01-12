import random
from typing import Any

from src.context import SemanticEntropyContext


def response_generator(context: SemanticEntropyContext, prompt: str) -> Any:
    """
    Generate a response from the model with logprobs for semantic entropy calculation.

    Args:
        context: SemanticEntropyContext containing model client and configuration.
        prompt: The input prompt to generate a response for.

    Returns:
        The model response object with logprobs.
    """
    if context.model_name == "llama3-instruct":
        if context.one_shot and context.prompts:
            # One-shot mode with random prompt selection
            response = context.model_client.model_predict(
                data=random.choice(context.prompts)
            )
        else:
            # Standard prediction
            response = context.model_client.model_predict(data=[prompt])
    else:
        # OpenAI-compatible API (GPT models)
        if context.one_shot and context.prompts:
            messages = [{"role": "user", "content": random.choice(context.prompts)}]
        elif context.messages:
            messages = [
                {"role": m["role"], "content": m["content"]} for m in context.messages
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = context.model_client.chat.completions.create(
            model=context.model_name,
            messages=messages,
            stream=False,
            logprobs=True,
        )

    return response
