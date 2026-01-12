"""Prompt/system leakage test.

Tests if a model reveals its system prompt or confidential instructions
through direct requests, override attempts, or indirect extraction.
"""

import logging

from services.evaluation.benchmarks.privacy.src.prompts import (
    TEST_SYSTEM_PROMPTS,
    get_leakage_attacks,
)
from services.evaluation.benchmarks.privacy.src.tests.base import (
    BasePrivacyTest,
    PrivacySample,
)

logger = logging.getLogger(__name__)


class PromptLeakageTest(BasePrivacyTest):
    """Tests for system prompt leakage vulnerabilities."""

    name = "prompt_leakage"
    description = "Tests if model reveals confidential instructions"

    def run(self, model) -> dict:
        """Run prompt leakage test.

        Args:
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(f"Running prompt leakage test with {self.num_samples} samples")

        samples: list[PrivacySample] = []
        category_results: dict[str, dict[str, int]] = {}

        # Get attacks by category
        categories = [
            "direct",
            "override",
            "summarization",
            "encoding",
            "role_confusion",
            "indirect",
        ]

        # Use provided system prompt or test prompts
        system_prompts_to_test = self._get_system_prompts_to_test()

        for system_prompt_info in system_prompts_to_test:
            prompt_name = system_prompt_info["name"]
            system_prompt = system_prompt_info["prompt"]

            # Temporarily set the system prompt for analysis
            original_system_prompt = self.system_prompt
            self.system_prompt = system_prompt

            for category in categories:
                attacks = get_leakage_attacks(category)
                if not attacks:
                    continue

                # Limit attacks per category
                attacks_to_use = attacks[: self.num_samples]

                for attack in attacks_to_use:
                    attack_name = attack["name"]
                    attack_template = attack["template"]

                    # Get model response with system prompt
                    response = self._get_model_response_with_system(
                        model, attack_template, system_prompt
                    )

                    # Analyze for instruction leakage
                    leakage_result = self._analyze_response(
                        response=response,
                        check_pii=False,
                        check_instructions=True,
                        check_confidence=False,
                    )

                    # Create sample
                    sample = self._create_sample(
                        probe_name=f"{prompt_name}_{attack_name}",
                        probe_category=category,
                        prompt=attack_template,
                        response=response,
                        leakage_result=leakage_result,
                        metadata={
                            "system_prompt_name": prompt_name,
                            "attack_description": attack.get("description", ""),
                            "system_prompt_length": len(system_prompt),
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

            # Restore original system prompt
            self.system_prompt = original_system_prompt

        # Calculate score (higher = safer)
        score = self._calculate_safety_score(samples)

        # Calculate category vulnerability
        category_vulnerability = {}
        for category, counts in category_results.items():
            total = counts["safe"] + counts["leaked"]
            if total > 0:
                category_vulnerability[category] = counts["leaked"] / total

        # Calculate attack effectiveness
        attack_effectiveness = self._calculate_attack_effectiveness(samples)

        return self._score_to_dict(
            score=score,
            samples=samples,
            extra_details={
                "category_vulnerability": category_vulnerability,
                "attack_effectiveness": attack_effectiveness,
                "most_effective_attack": (
                    max(attack_effectiveness, key=attack_effectiveness.get)
                    if attack_effectiveness
                    else None
                ),
                "most_vulnerable_category": (
                    max(category_vulnerability, key=category_vulnerability.get)
                    if category_vulnerability
                    else None
                ),
                "system_prompts_tested": len(system_prompts_to_test),
            },
        )

    def _get_system_prompts_to_test(self) -> list[dict]:
        """Get system prompts to test.

        Uses provided system prompt or falls back to test prompts.

        Returns:
            List of system prompt dictionaries.
        """
        if self.system_prompt:
            return [
                {
                    "name": "provided",
                    "prompt": self.system_prompt,
                }
            ]

        # Use first few test prompts
        return TEST_SYSTEM_PROMPTS[: max(1, self.num_samples // 2)]

    def _calculate_attack_effectiveness(
        self, samples: list[PrivacySample]
    ) -> dict[str, float]:
        """Calculate effectiveness of each attack type.

        Args:
            samples: List of test samples.

        Returns:
            Dictionary mapping attack names to success rates.
        """
        attack_counts: dict[str, dict[str, int]] = {}

        for sample in samples:
            # Extract attack name from probe name
            parts = sample.probe_name.split("_", 1)
            attack_name = parts[1] if len(parts) > 1 else sample.probe_name

            if attack_name not in attack_counts:
                attack_counts[attack_name] = {"success": 0, "total": 0}

            attack_counts[attack_name]["total"] += 1
            if not sample.is_safe:  # Leakage = attack success
                attack_counts[attack_name]["success"] += 1

        # Calculate success rates
        effectiveness = {}
        for attack_name, counts in attack_counts.items():
            if counts["total"] > 0:
                effectiveness[attack_name] = counts["success"] / counts["total"]

        return effectiveness
