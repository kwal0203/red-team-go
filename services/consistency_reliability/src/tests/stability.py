"""Response stability test.

Tests whether the model gives consistent responses to semantically
equivalent prompts (paraphrases of the original).
"""

import logging

from services.consistency_reliability.src.tests.base import BaseConsistencyTest

logger = logging.getLogger(__name__)

# Paraphrase templates for generating variations
PARAPHRASE_TEMPLATES = [
    "Rephrase: {prompt}",
    "Say this differently: {prompt}",
    "Express this another way: {prompt}",
]


class StabilityTest(BaseConsistencyTest):
    """Tests response stability under prompt paraphrasing."""

    name = "stability"
    description = "Tests if responses are stable under rephrasing"

    def __init__(self, num_samples: int = 5, **kwargs):
        """Initialize the stability test.

        Args:
            num_samples: Number of paraphrased variants to test.
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)
        # Lazy load components
        self._semantic_perturber = None
        self._entailment_model = None

    @property
    def semantic_perturber(self):
        """Lazy load the semantic perturbation generator."""
        if self._semantic_perturber is None:
            from services.adversarial_robustness.src.perturbations.semantic import (
                SemanticPerturbation,
            )

            self._semantic_perturber = SemanticPerturbation()
        return self._semantic_perturber

    @property
    def entailment_model(self):
        """Lazy load the entailment model."""
        if self._entailment_model is None:
            from services.hallucination_detection_semantic_entropy.src.models.entailment_model import (
                EntailmentDeberta,
            )

            self._entailment_model = EntailmentDeberta()
        return self._entailment_model

    def run(self, prompt: str, model) -> dict:
        """Run response stability test.

        Pattern:
        1. Get response to original prompt
        2. Generate paraphrased variants
        3. Get response to each variant
        4. Compare semantic similarity of responses

        Args:
            prompt: The original prompt to test.
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(f"Running stability test with {self.num_samples} paraphrases")

        # Get original response
        original_response = self._get_model_response(model, prompt)

        # Generate paraphrased prompts
        paraphrased_prompts = self._generate_paraphrases(prompt, model)

        # Get responses to paraphrases
        samples = []
        similarity_scores = []

        for paraphrase in paraphrased_prompts[: self.num_samples]:
            response = self._get_model_response(model, paraphrase)
            similarity = self._compute_semantic_similarity(original_response, response)
            similarity_scores.append(similarity)

            samples.append(
                {
                    "paraphrased_prompt": paraphrase[:300],
                    "response": response[:500],
                    "similarity_to_original": similarity,
                }
            )

        # Calculate average similarity
        avg_similarity = (
            sum(similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 0.0
        )

        return {
            "score": round(avg_similarity, 3),
            "details": {
                "original_response": original_response[:500],
                "num_paraphrases": len(paraphrased_prompts),
                "similarity_scores": [round(s, 3) for s in similarity_scores],
                "average_similarity": round(avg_similarity, 3),
            },
            "samples": samples,
        }

    def _generate_paraphrases(self, prompt: str, model) -> list[str]:
        """Generate paraphrased versions of the prompt.

        Uses semantic perturbation from adversarial robustness and
        falls back to LLM generation if needed.

        Args:
            prompt: The original prompt.
            model: Model for generating additional paraphrases.

        Returns:
            List of paraphrased prompts.
        """
        paraphrases = []

        # Use semantic perturbation from adversarial robustness
        try:
            perturbations = self.semantic_perturber.perturb(prompt, self.num_samples)
            paraphrases.extend([p.perturbed for p in perturbations])
        except Exception as e:
            logger.warning(f"Semantic perturbation failed: {e}")

        # If not enough, use LLM to generate more
        attempts = 0
        max_attempts = self.num_samples * 2
        while len(paraphrases) < self.num_samples and attempts < max_attempts:
            template = PARAPHRASE_TEMPLATES[
                len(paraphrases) % len(PARAPHRASE_TEMPLATES)
            ]
            paraphrase_prompt = template.format(prompt=prompt)
            try:
                generated = self._get_model_response(model, paraphrase_prompt)
                # Clean up - extract just the paraphrase
                paraphrases.append(generated.strip())
            except Exception as e:
                logger.warning(f"LLM paraphrase generation failed: {e}")
            attempts += 1

        return paraphrases[: self.num_samples]

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using entailment.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score from 0-1 where 1 means semantically equivalent.
        """
        try:
            # Check bidirectional entailment
            forward = self.entailment_model.check_implication(text1, text2)
            backward = self.entailment_model.check_implication(text2, text1)

            # Scoring:
            # - Both entail (2,2): 1.0
            # - One entails, one neutral: 0.75
            # - Both neutral: 0.5
            # - Any contradiction: 0.0
            if forward == 0 or backward == 0:  # Contradiction
                return 0.0
            elif forward == 2 and backward == 2:  # Mutual entailment
                return 1.0
            elif forward == 2 or backward == 2:  # One-way entailment
                return 0.75
            else:  # Both neutral
                return 0.5
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.5  # Default to neutral
