"""Adapters for experimental red teaming prototypes.

Each adapter maps an experimental method into the BaseGenerator interface. The
actual research code lives under the /experimental directory; these classes
provide a single registry entry per method and surface where the prototype
resides so it can be wired into the execution layer.

For now these adapters surface clear NotImplemented errors that point to the
prototype entrypoint. They can be replaced incrementally with real adapters
that translate prototype outputs into GeneratedPrompt/PromptArtifact objects.
"""

import logging

from services.prompt_generation.src.generators.base import BaseGenerator

logger = logging.getLogger(__name__)


EXPERIMENTAL_METHODS: dict[str, dict[str, str]] = {
    "stp": {
        "path": "experimental/single/scalable-and-transferable/main.py",
        "description": "STP: structured jailbreak artifacts (instruction/persona/modulation)",
    },
    "dsn": {
        "path": "experimental/single/dont-say-no/main.py",
        "description": "DSN: gradient-based refusal suppression suffix",
    },
    "manyshot": {
        "path": "experimental/single/many-shot-jailbreak/main.py",
        "description": "Many-shot: long ICL jailbreak transcript construction",
    },
    "advprompter": {
        "path": "experimental/iterative/adv-prompt/main.py",
        "description": "AdvPrompter: stochastic beam search adversarial suffixes",
    },
    "autodan": {
        "path": "experimental/iterative/autodan/main.py",
        "description": "AutoDAN: genetic algorithm prompt evolution",
    },
    "cold": {
        "path": "experimental/iterative/cold-attack/main.py",
        "description": "COLD-Attack: soft-prompt optimization with energy minimization",
    },
    "crt": {
        "path": "experimental/iterative/curiosity-driven/main.py",
        "description": "Curiosity-Driven Red Teaming: PPO-style exploration",
    },
    "gptfuzzer": {
        "path": "experimental/iterative/gpt-fuzzer/main.py",
        "description": "GPTFUZZER: MCTS-based template mutation",
    },
    "jailbreakhub": {
        "path": "experimental/iterative/do-anything/main.py",
        "description": "JAILBREAKHUB: clustering/ASR analytics for in-the-wild prompts",
    },
    "blackbox_pair": {
        "path": "experimental/iterative/black-box/main.py",
        "description": "PAIR (black-box): attacker/target/judge stream refinement",
    },
    "datasets": {
        "path": "experimental/single/datasets/main.py",
        "description": "Placeholder dataset generator",
    },
}


class ExperimentalPromptGenerator(BaseGenerator):
    """Base adapter that points to a specific experimental prototype."""

    method_id: str = ""

    def generate(
        self,
        category: str,
        num_prompts: int = 10,
        seed_prompt: str | None = None,
    ):
        """Raise a clear placeholder until the prototype is wired."""
        method_info = EXPERIMENTAL_METHODS.get(self.method_id, {})
        path_hint = method_info.get("path", "experimental")
        raise NotImplementedError(
            f"{self.method_id} prototype not yet wired into the service. "
            f"Run the prototype at {path_hint} and wrap its outputs into "
            "GeneratedPrompt/PromptArtifact to enable this generator."
        )


class DSNPromptGenerator(ExperimentalPromptGenerator):
    method_id = "dsn"


class ManyShotPromptGenerator(ExperimentalPromptGenerator):
    method_id = "manyshot"


class AdvPrompterGenerator(ExperimentalPromptGenerator):
    method_id = "advprompter"


class AutoDANPromptGenerator(ExperimentalPromptGenerator):
    method_id = "autodan"


class ColdAttackPromptGenerator(ExperimentalPromptGenerator):
    method_id = "cold"


class CRTPromptGenerator(ExperimentalPromptGenerator):
    method_id = "crt"


class GPTFuzzerPromptGenerator(ExperimentalPromptGenerator):
    method_id = "gptfuzzer"


class JailbreakHubGenerator(ExperimentalPromptGenerator):
    method_id = "jailbreakhub"


class BlackBoxPAIRPromptGenerator(ExperimentalPromptGenerator):
    method_id = "blackbox_pair"


class DatasetPromptGenerator(ExperimentalPromptGenerator):
    method_id = "datasets"


class STPPromptGenerator(ExperimentalPromptGenerator):
    method_id = "stp"

    def generate(
        self,
        category: str,
        num_prompts: int = 1,
        seed_prompt: str | None = None,
    ):
        """Call the STP prototype to synthesize structured artifacts."""
        from services.prompt_generation.src.generators.stp_adapter import (
            run_stp_once,
        )

        artifacts = run_stp_once(category=category, seed_prompt=seed_prompt)
        results = []
        for _idx, artifact in enumerate(artifacts[:num_prompts]):
            results.append(artifact)
            if len(results) >= num_prompts:
                break
        return results
