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
    artifact: "PromptArtifact | None" = None
    metrics: "GenerationMetrics | None" = None
    run_metadata: "RunMetadata | None" = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "generation_method": self.generation_method,
            "metadata": self.metadata,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "run_metadata": self.run_metadata.to_dict() if self.run_metadata else None,
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


@dataclass
class PromptExample:
    """A single in-context learning example."""

    user: str
    assistant: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"user": self.user, "assistant": self.assistant}


@dataclass
class PromptArtifact:
    """Structured jailbreak artifact components."""

    system: str | None = None
    instruction: str | None = None
    persona: str | None = None
    suffix: str | None = None
    icl_examples: list[PromptExample] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "system": self.system,
            "instruction": self.instruction,
            "persona": self.persona,
            "suffix": self.suffix,
            "icl_examples": [ex.to_dict() for ex in self.icl_examples],
        }


@dataclass
class GenerationMetrics:
    """Evaluation metrics returned by iterative/optimized generators."""

    asr: float | None = None
    judge_score: float | None = None
    refusal_rate: float | None = None
    novelty: float | None = None
    cost: float | None = None
    tokens: int | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "asr": self.asr,
            "judge_score": self.judge_score,
            "refusal_rate": self.refusal_rate,
            "novelty": self.novelty,
            "cost": self.cost,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class RunMetadata:
    """Metadata describing how/where a generator executed."""

    method: str
    backend: str | None = None
    run_id: str | None = None
    logs_path: str | None = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "backend": self.backend,
            "run_id": self.run_id,
            "logs_path": self.logs_path,
            "config": self.config,
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
