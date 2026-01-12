"""Semantic-level text perturbations.

Includes paraphrase attacks that preserve meaning while changing surface form.
Currently stubbed for LLM-based paraphrasing.
"""

from services.evaluation.benchmarks.robustness.src.perturbations.base import (
    BasePerturbation,
    PerturbedText,
)

# Template-based paraphrase patterns
PARAPHRASE_TEMPLATES = [
    ("How do I", "What is the process for"),
    ("How do I", "Can you explain how to"),
    ("How do I", "What steps should I take to"),
    ("How can I", "What is the best way to"),
    ("How can I", "Is there a method to"),
    ("Tell me how to", "Explain the procedure for"),
    ("Tell me how to", "Describe how one would"),
    ("I want to", "I am interested in"),
    ("I need to", "I require assistance with"),
    ("Help me", "Assist me in"),
    ("Show me", "Demonstrate"),
    ("Give me", "Provide"),
]


class SemanticPerturbation(BasePerturbation):
    """Semantic-level perturbations that preserve meaning.

    Currently uses template-based paraphrasing. LLM-based paraphrasing
    can be added for more sophisticated semantic transformations.
    """

    name = "semantic"
    category = "semantic"

    def perturb(self, text: str, num_variants: int = 5) -> list[PerturbedText]:
        """Generate semantic perturbations using templates.

        For more sophisticated paraphrasing, use LLM-based methods.

        Args:
            text: Original text to perturb.
            num_variants: Number of variants to generate.

        Returns:
            List of perturbed text variants.
        """
        results: list[PerturbedText] = []

        # Template-based paraphrasing
        results.extend(self._template_paraphrase(text, num_variants))

        return results

    def _template_paraphrase(self, text: str, num_variants: int) -> list[PerturbedText]:
        """Apply template-based paraphrase transformations."""
        results = []
        text_lower = text.lower()

        for pattern, replacement in PARAPHRASE_TEMPLATES:
            if pattern.lower() in text_lower:
                # Find the pattern case-insensitively and replace
                import re

                perturbed = re.sub(
                    re.escape(pattern),
                    replacement,
                    text,
                    flags=re.IGNORECASE,
                    count=1,
                )

                if perturbed != text:
                    results.append(
                        self._create_result(
                            original=text,
                            perturbed=perturbed,
                            method="template_paraphrase",
                            changes=[f"'{pattern}'→'{replacement}'"],
                        )
                    )

                if len(results) >= num_variants:
                    break

        return results[:num_variants]

    def paraphrase_with_llm(
        self, text: str, model, num_variants: int = 5
    ) -> list[PerturbedText]:
        """Generate paraphrases using an LLM.

        This method uses the target model to generate semantically
        equivalent but syntactically different versions of the input.

        Args:
            text: Original text to paraphrase.
            model: LLM model wrapper to use for generation.
            num_variants: Number of paraphrases to generate.

        Returns:
            List of perturbed text variants.

        Raises:
            NotImplementedError: LLM paraphrasing requires model integration.
        """
        raise NotImplementedError(
            "LLM-based paraphrasing not yet implemented.\n"
            "To enable, provide an LLM model wrapper that can generate paraphrases."
        )
