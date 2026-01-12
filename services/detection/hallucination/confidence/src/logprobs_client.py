"""Client for extracting logprobs from language model APIs."""

import logging
from dataclasses import dataclass

import openai

from utils.config import get_openai_key

logger = logging.getLogger(__name__)


@dataclass
class TokenLogprob:
    """Log probability information for a single token."""

    token: str
    logprob: float
    top_logprobs: list[dict[str, float]] | None = None


@dataclass
class LogprobsResponse:
    """Response containing text and token-level log probabilities."""

    text: str
    tokens: list[TokenLogprob]
    total_tokens: int
    model: str

    @property
    def logprobs(self) -> list[float]:
        """Return just the logprob values."""
        return [t.logprob for t in self.tokens]


class LogprobsClient:
    """Client for getting model responses with log probabilities.

    Currently supports OpenAI models. Can be extended for other providers.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo") -> None:
        """Initialize the logprobs client.

        Args:
            model_name: Name of the model to use for generation.
        """
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=get_openai_key())
        logger.info(f"LogprobsClient initialized with model: {model_name}")

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_logprobs: int | None = None,
    ) -> LogprobsResponse:
        """Generate text and return token-level log probabilities.

        Args:
            prompt: The user prompt to send to the model.
            system_prompt: Optional system prompt for context.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for deterministic).
            top_logprobs: Number of top logprobs to return per token (1-20).

        Returns:
            LogprobsResponse containing generated text and token logprobs.

        Raises:
            ValueError: If API call fails or logprobs not available.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating with logprobs for prompt: {prompt[:100]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ValueError(f"Failed to get logprobs from API: {e}") from e

        choice = response.choices[0]
        content = choice.message.content or ""

        if not choice.logprobs or not choice.logprobs.content:
            raise ValueError(
                f"Model {self.model_name} did not return logprobs. "
                "Ensure the model supports logprobs parameter."
            )

        tokens = []
        for token_info in choice.logprobs.content:
            top_probs = None
            if token_info.top_logprobs:
                top_probs = [{tp.token: tp.logprob} for tp in token_info.top_logprobs]

            tokens.append(
                TokenLogprob(
                    token=token_info.token,
                    logprob=token_info.logprob,
                    top_logprobs=top_probs,
                )
            )

        logger.debug(f"Generated {len(tokens)} tokens with logprobs")

        return LogprobsResponse(
            text=content,
            tokens=tokens,
            total_tokens=len(tokens),
            model=self.model_name,
        )

    def get_completion_logprobs(
        self,
        prompt: str,
        completion: str,
        system_prompt: str | None = None,
    ) -> LogprobsResponse:
        """Get logprobs for a specific completion given a prompt.

        This generates a new completion and returns its logprobs.
        For evaluating pre-existing completions, use the echo parameter
        (not available in chat models).

        Args:
            prompt: The input prompt.
            completion: The completion to evaluate (used as reference).
            system_prompt: Optional system prompt.

        Returns:
            LogprobsResponse for the model's completion.

        Note:
            The returned completion may differ from the input completion.
            This is a limitation of chat-based models.
        """
        # For chat models, we can't directly get logprobs of a specific completion
        # We generate a new completion and return its logprobs
        # This is suitable for evaluating model confidence in its own generations
        return self.generate_with_logprobs(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max(len(completion.split()) * 2, 100),
            temperature=0.0,
        )
