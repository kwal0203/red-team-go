"""Base class for consistency and reliability tests."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """A single evaluation sample with input and output."""

    input_prompt: str
    model_response: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_prompt": self.input_prompt,
            "model_response": self.model_response,
            "metadata": self.metadata,
        }


class BaseConsistencyTest(ABC):
    """Abstract base class for consistency tests.

    All consistency/reliability tests should inherit from this class
    and implement the run method.
    """

    name: str = "base"
    description: str = "Base consistency test"

    def __init__(self, num_samples: int = 5, **kwargs):
        """Initialize the test.

        Args:
            num_samples: Number of samples/variations to generate.
            **kwargs: Test-specific configuration.
        """
        self.num_samples = num_samples
        self.config = kwargs

    @abstractmethod
    def run(self, prompt: str, model) -> dict:
        """Run the test on a prompt with the given model.

        Args:
            prompt: The prompt to test.
            model: Target model wrapper with model_predict method.

        Returns:
            Dictionary containing:
            - score: float from 0-1
            - details: test-specific details
            - samples: optional list of per-sample results
        """
        pass

    def _get_model_response(self, model, prompt: str) -> str:
        """Helper to get a single response from the model."""
        response = model.model_predict(data=[prompt])
        if isinstance(response, list):
            return response[0]
        return response

    def _get_multiple_responses(
        self, model, prompt: str, n: int, temperature: float = 0.9
    ) -> list[str]:
        """Helper to get multiple responses for the same prompt.

        Note: Temperature control may need model-specific implementation.
        For now, relies on calling model_predict multiple times.
        """
        responses = []
        for _ in range(n):
            response = self._get_model_response(model, prompt)
            responses.append(response)
        return responses
