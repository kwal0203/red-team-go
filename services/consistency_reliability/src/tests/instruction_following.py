"""Instruction following fidelity test.

Tests whether the model correctly follows specific formatting
and constraint instructions.
"""

import logging
import re
from collections.abc import Callable

from services.consistency_reliability.src.tests.base import BaseConsistencyTest

logger = logging.getLogger(__name__)

# Default instruction constraints with validation functions
DEFAULT_CONSTRAINTS = [
    {
        "id": "bullet_points_3",
        "instruction": "Respond using exactly 3 bullet points.",
        "validator": lambda r: len(re.findall(r"^[\s]*[-*•]", r, re.MULTILINE)) == 3,
    },
    {
        "id": "lowercase_only",
        "instruction": "Respond using only lowercase letters.",
        "validator": lambda r: r == r.lower() or not any(c.isalpha() for c in r),
    },
    {
        "id": "word_limit_50",
        "instruction": "Respond in 50 words or fewer.",
        "validator": lambda r: len(r.split()) <= 50,
    },
    {
        "id": "numbered_list",
        "instruction": "Respond with a numbered list (1., 2., 3., etc.).",
        "validator": lambda r: bool(re.search(r"^\s*1\.", r, re.MULTILINE)),
    },
    {
        "id": "no_questions",
        "instruction": "Do not ask any questions in your response.",
        "validator": lambda r: "?" not in r,
    },
    {
        "id": "start_with_therefore",
        "instruction": "Start your response with the word 'Therefore'.",
        "validator": lambda r: r.strip().lower().startswith("therefore"),
    },
    {
        "id": "single_sentence",
        "instruction": "Respond in exactly one sentence.",
        "validator": lambda r: r.count(".") == 1 and r.strip().endswith("."),
    },
    {
        "id": "all_caps",
        "instruction": "Respond using only uppercase letters.",
        "validator": lambda r: r == r.upper() or not any(c.isalpha() for c in r),
    },
]


class InstructionFollowingTest(BaseConsistencyTest):
    """Tests instruction following fidelity."""

    name = "instruction_following"
    description = "Tests if model follows specific formatting constraints"

    def __init__(
        self,
        num_samples: int = 5,
        instruction_constraints: list[str] | None = None,
        **kwargs,
    ):
        """Initialize the instruction following test.

        Args:
            num_samples: Number of constraints to test.
            instruction_constraints: Custom constraint instructions.
            **kwargs: Additional configuration.
        """
        super().__init__(num_samples=num_samples, **kwargs)

        # Parse custom constraints or use defaults
        if instruction_constraints:
            self.constraints = self._parse_custom_constraints(instruction_constraints)
        else:
            self.constraints = DEFAULT_CONSTRAINTS[: self.num_samples]

    def _parse_custom_constraints(self, custom: list[str]) -> list[dict]:
        """Parse custom constraint strings into constraint objects.

        Custom constraints use a simple format. For complex validation,
        use the default constraints.

        Args:
            custom: List of constraint instruction strings.

        Returns:
            List of constraint dictionaries with validators.
        """
        parsed = []
        for i, constraint in enumerate(custom):
            parsed.append(
                {
                    "id": f"custom_{i}",
                    "instruction": constraint,
                    "validator": self._create_simple_validator(constraint),
                }
            )
        return parsed

    def _create_simple_validator(self, instruction: str) -> Callable[[str], bool]:
        """Create a simple validator based on instruction text.

        This is a best-effort approach for custom constraints.

        Args:
            instruction: The instruction text.

        Returns:
            A validator function.
        """
        instruction_lower = instruction.lower()

        # Pattern matching for common constraint types
        if "bullet" in instruction_lower:
            match = re.search(r"(\d+)", instruction)
            n = int(match.group(1)) if match else 3
            return lambda r, n=n: len(re.findall(r"^[\s]*[-*•]", r, re.MULTILINE)) == n

        if "lowercase" in instruction_lower:
            return lambda r: r == r.lower() or not any(c.isalpha() for c in r)

        if "uppercase" in instruction_lower:
            return lambda r: r == r.upper() or not any(c.isalpha() for c in r)

        if "word" in instruction_lower and (
            "fewer" in instruction_lower or "less" in instruction_lower
        ):
            match = re.search(r"(\d+)", instruction)
            n = int(match.group(1)) if match else 50
            return lambda r, n=n: len(r.split()) <= n

        # Default: always pass (can't validate unknown constraint)
        logger.warning(f"Could not create validator for: {instruction}")
        return lambda r: True

    def run(self, prompt: str, model) -> dict:
        """Run instruction following test.

        Pattern:
        1. Append each instruction constraint to the prompt
        2. Get model response
        3. Check if constraint is satisfied

        Args:
            prompt: The base prompt to test with.
            model: Target model wrapper.

        Returns:
            Dictionary with score, details, and samples.
        """
        logger.info(
            f"Running instruction following test with {len(self.constraints)} constraints"
        )

        samples = []
        passed_count = 0

        for constraint in self.constraints[: self.num_samples]:
            # Create prompt with instruction
            constrained_prompt = f"{prompt}\n\n{constraint['instruction']}"

            # Get response
            response = self._get_model_response(model, constrained_prompt)

            # Validate
            try:
                passed = constraint["validator"](response)
            except Exception as e:
                logger.warning(f"Validator failed for {constraint['id']}: {e}")
                passed = False

            if passed:
                passed_count += 1

            samples.append(
                {
                    "constraint_id": constraint["id"],
                    "instruction": constraint["instruction"],
                    "response": response[:500],
                    "passed": passed,
                }
            )

        # Calculate compliance rate
        compliance_rate = passed_count / len(samples) if samples else 0.0

        return {
            "score": round(compliance_rate, 3),
            "details": {
                "instructions_tested": [
                    c["instruction"] for c in self.constraints[: self.num_samples]
                ],
                "passed_count": passed_count,
                "total_count": len(samples),
                "compliance_rate": round(compliance_rate, 3),
            },
            "samples": samples,
        }
