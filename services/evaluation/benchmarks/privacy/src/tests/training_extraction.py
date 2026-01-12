"""Training data extraction test.

Tests if a model leaks memorized training data through various
extraction techniques including repetition attacks, prefix completion,
and verbatim probing.
"""

import logging

from services.evaluation.benchmarks.privacy.src.prompts import (
    format_probe,
    get_extraction_probes,
)
from services.evaluation.benchmarks.privacy.src.tests.base import (
    BasePrivacyTest,
    PrivacySample,
)

logger = logging.getLogger(__name__)


class TrainingDataExtractionTest(BasePrivacyTest):
    """Tests for training data memorization and extraction."""

    name = "training_extraction"
    description = "Tests if model leaks memorized training data"

    def run(self, model) -> dict:
        """Run training data extraction test.

        Args:
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(
            f"Running training data extraction test with {self.num_samples} samples"
        )

        samples: list[PrivacySample] = []
        category_results: dict[str, dict[str, int]] = {}

        # Get probes by category
        categories = ["repetition", "prefix_completion", "verbatim", "continuation"]

        for category in categories:
            probes = get_extraction_probes(category)
            if not probes:
                continue

            # Limit probes per category
            probes_to_use = probes[: self.num_samples]

            for probe in probes_to_use:
                probe_name = probe["name"]
                template = probe["template"]

                # Format the probe with test values
                prompt = format_probe(template, self.config.get("probe_values"))

                # Get model response
                response = self._get_model_response(model, prompt)

                # Analyze for leakage - focus on PII for extraction tests
                leakage_result = self._analyze_response(
                    response=response,
                    check_pii=True,
                    check_instructions=False,
                    check_confidence=True,
                    test_strings=self.config.get("test_strings"),
                )

                # Create sample
                sample = self._create_sample(
                    probe_name=probe_name,
                    probe_category=category,
                    prompt=prompt,
                    response=response,
                    leakage_result=leakage_result,
                    metadata={
                        "template": template,
                        "description": probe.get("description", ""),
                    },
                )
                samples.append(sample)

                # Track category results
                if category not in category_results:
                    category_results[category] = {"safe": 0, "leaked": 0}
                if sample.is_safe:
                    category_results[category]["safe"] += 1
                else:
                    category_results[category]["leaked"] += 1

        # Calculate score (higher = safer)
        score = self._calculate_safety_score(samples)

        # Calculate category vulnerability
        category_vulnerability = {}
        for category, counts in category_results.items():
            total = counts["safe"] + counts["leaked"]
            if total > 0:
                # Higher = more vulnerable (more leakages)
                category_vulnerability[category] = counts["leaked"] / total

        return self._score_to_dict(
            score=score,
            samples=samples,
            extra_details={
                "category_vulnerability": category_vulnerability,
                "most_vulnerable_category": (
                    max(category_vulnerability, key=category_vulnerability.get)
                    if category_vulnerability
                    else None
                ),
                "least_vulnerable_category": (
                    min(category_vulnerability, key=category_vulnerability.get)
                    if category_vulnerability
                    else None
                ),
                "probes_by_category": {
                    cat: len(get_extraction_probes(cat)) for cat in categories
                },
            },
        )
