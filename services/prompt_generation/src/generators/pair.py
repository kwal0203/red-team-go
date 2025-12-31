"""PAIR (Prompt Automatic Iterative Refinement) generator.

Uses iterative refinement between attacker and target models.
Currently stubbed for future implementation.

Reference:
Chao, P., et al. (2023). "Jailbreaking Black Box Large Language Models in Twenty Queries."
arXiv:2310.08419
"""

from services.prompt_generation.src.generators.base import (
    BaseGenerator,
    GeneratedPrompt,
)


class PAIRPromptGenerator(BaseGenerator):
    """Generates adversarial prompts using PAIR methodology.

    PAIR (Prompt Automatic Iterative Refinement) works by:
    1. Attacker LLM generates an initial jailbreak prompt
    2. Target LLM responds to the prompt
    3. Attacker analyzes response and refines the prompt
    4. Iterate until successful jailbreak or max iterations

    This is an effective method for black-box jailbreaking that
    typically succeeds within 20 queries.

    NOT YET IMPLEMENTED.
    """

    name = "pair"
    description = "PAIR: Prompt Automatic Iterative Refinement"

    def generate(
        self,
        category: str,
        num_prompts: int = 10,
        seed_prompt: str | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate adversarial prompts using PAIR.

        Args:
            category: Target category (e.g., "jailbreak", "harmful", "bias").
            num_prompts: Number of prompts to generate.
            seed_prompt: Optional seed prompt to start refinement.

        Returns:
            List of GeneratedPrompt objects.

        Raises:
            NotImplementedError: This generator is not yet implemented.
        """
        raise NotImplementedError(
            "PAIR prompt generator not yet implemented.\n\n"
            "PAIR (Prompt Automatic Iterative Refinement) methodology:\n"
            "1. Attacker LLM generates initial jailbreak prompt\n"
            "2. Target LLM responds\n"
            "3. Attacker analyzes response and refines\n"
            "4. Iterate until success (typically < 20 queries)\n\n"
            "Reference:\n"
            "Chao et al. (2023). 'Jailbreaking Black Box Large Language Models\n"
            "in Twenty Queries.' arXiv:2310.08419\n\n"
            "Use the 'llm' generator for now."
        )
