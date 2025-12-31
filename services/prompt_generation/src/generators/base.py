"""Base classes for prompt generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GeneratedPrompt:
    """A generated adversarial prompt.

    Attributes:
        prompt: The generated prompt text.
        generation_method: Method used to generate this prompt.
        metadata: Additional metadata about generation.
    """

    prompt: str
    generation_method: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "generation_method": self.generation_method,
            "metadata": self.metadata,
        }


@dataclass
class PromptEvaluation:
    """Evaluation result for a generated prompt.

    Attributes:
        prompt: The prompt that was evaluated.
        model_response: Response from the target model.
        bypass_successful: Whether the prompt bypassed safety.
        guardrail_results: Detailed guardrail check results.
    """

    prompt: str
    model_response: str
    bypass_successful: bool
    guardrail_results: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "model_response": self.model_response,
            "bypass_successful": self.bypass_successful,
            "guardrail_results": self.guardrail_results,
        }


class BaseGenerator(ABC):
    """Abstract base class for prompt generators.

    Generators create adversarial prompts for red-teaming LLMs.
    """

    name: str = "base"
    description: str = "Base generator"

    @abstractmethod
    def generate(
        self,
        category: str,
        num_prompts: int = 10,
        seed_prompt: str | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate adversarial prompts.

        Args:
            category: Target category (e.g., "jailbreak", "harmful", "bias").
            num_prompts: Number of prompts to generate.
            seed_prompt: Optional seed prompt to base generation on.

        Returns:
            List of GeneratedPrompt objects.
        """
        pass
