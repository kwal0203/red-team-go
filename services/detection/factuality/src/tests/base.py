"""Base class for misinformation and factuality tests."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FactualitySample:
    """A single factuality test sample with input and output."""

    input_prompt: str
    model_response: str
    expected_answer: str | None = None
    is_correct: bool | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "input_prompt": self.input_prompt,
            "model_response": self.model_response,
            "expected_answer": self.expected_answer,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseFactualityTest(ABC):
    """Abstract base class for factuality tests.

    All misinformation/factuality tests should inherit from this class
    and implement the run method.
    """

    name: str = "base"
    description: str = "Base factuality test"

    def __init__(self, num_samples: int = 5, **kwargs):
        """Initialize the test.

        Args:
            num_samples: Number of samples/questions to test.
            **kwargs: Test-specific configuration.
        """
        self.num_samples = num_samples
        self.config = kwargs

    @abstractmethod
    def run(self, prompt: str, model) -> dict:
        """Run the test on a prompt with the given model.

        Args:
            prompt: The base prompt/topic to test.
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

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        return answer.strip().lower()

    def _check_answer_contains(self, response: str, expected: str) -> bool:
        """Check if response contains the expected answer."""
        return self._normalize_answer(expected) in self._normalize_answer(response)
