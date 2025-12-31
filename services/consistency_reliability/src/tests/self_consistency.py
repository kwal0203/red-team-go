"""Self-consistency test.

Tests whether the model gives consistent answers when asked the same
question multiple times (with high temperature sampling).
"""

import logging

from services.consistency_reliability.src.tests.base import BaseConsistencyTest

logger = logging.getLogger(__name__)


class SelfConsistencyTest(BaseConsistencyTest):
    """Tests self-consistency across multiple generations."""

    name = "self_consistency"
    description = "Tests if model gives consistent answers to same question"

    def __init__(self, num_samples: int = 5, **kwargs):
        """Initialize the self-consistency test.

        Args:
            num_samples: Number of response samples to generate.
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)
        # Lazy load entailment model
        self._entailment_model = None

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
        """Run self-consistency test.

        Pattern:
        1. Ask the same question N times
        2. Cluster responses by semantic equivalence
        3. Score based on size of largest cluster

        Args:
            prompt: The prompt to test.
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(
            f"Running self-consistency test with {self.num_samples} generations"
        )

        # Generate multiple responses to the same prompt
        responses = self._get_multiple_responses(model, prompt, self.num_samples)

        # Cluster responses by semantic equivalence
        cluster_ids = self._cluster_responses(responses)

        # Calculate statistics
        num_clusters = len(set(cluster_ids))
        cluster_sizes: dict[int, int] = {}
        for cid in cluster_ids:
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
        largest_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 0

        # Score = ratio of responses in largest cluster
        consistency_ratio = largest_cluster_size / len(responses) if responses else 0.0

        samples = [
            {
                "response": resp[:500],
                "cluster_id": cid,
            }
            for resp, cid in zip(responses, cluster_ids, strict=False)
        ]

        return {
            "score": round(consistency_ratio, 3),
            "details": {
                "num_clusters": num_clusters,
                "largest_cluster_size": largest_cluster_size,
                "cluster_distribution": cluster_sizes,
                "consistency_ratio": round(consistency_ratio, 3),
            },
            "samples": samples,
        }

    def _cluster_responses(self, responses: list[str]) -> list[int]:
        """Cluster responses by semantic equivalence.

        Uses the semantic_ids logic from semantic entropy service.

        Args:
            responses: List of response texts.

        Returns:
            List of cluster IDs for each response.
        """
        from services.hallucination_detection_semantic_entropy.src.services.semantic_ids import (
            get_semantic_ids,
        )

        return get_semantic_ids(responses, self.entailment_model)
