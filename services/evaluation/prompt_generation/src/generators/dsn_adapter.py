"""Adapter for the DSN (Don't-Say-No) prototype.

This wraps the DSN service to generate refusal-suppression suffixes without
depending on the experimental prototype.
"""

import time
import uuid

from services.evaluation.attacks.dsn.api import DSNRequest
from services.evaluation.attacks.dsn.service import dsn_service
from services.evaluation.prompt_generation.src.generators.base import (
    BaseGenerator,
    GeneratedPrompt,
    GenerationMetrics,
    PromptArtifact,
    RunMetadata,
)


class DSNPromptGenerator(BaseGenerator):
    """Generates adversarial suffixes inspired by the DSN prototype."""

    name = "dsn"
    description = "DSN: gradient-style refusal suppression suffix generation"

    def generate(
        self,
        category: str,
        num_prompts: int = 3,
        seed_prompt: str | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate DSN-style suffix artifacts via the DSN service."""
        run_id = f"dsn-{uuid.uuid4().hex[:8]}"
        start = time.time()

        request = DSNRequest(
            seed_prompt=seed_prompt,
            num_suffixes=num_prompts,
        )
        response = dsn_service(request)

        outputs: list[GeneratedPrompt] = []
        for idx, suffix in enumerate(response.suffixes):
            artifact = PromptArtifact(
                system=None,
                instruction=None,
                persona=None,
                suffix=suffix.suffix,
                icl_examples=[],
            )

            metrics = GenerationMetrics(
                judge_score=None,
                refusal_rate=None,
                latency_ms=int((time.time() - start) * 1000),
            )

            run_meta = RunMetadata(
                method="dsn",
                backend=response.metadata.get("backend"),
                run_id=response.metadata.get("run_id", run_id),
                logs_path=None,
                config=response.metadata,
            )

            outputs.append(
                GeneratedPrompt(
                    prompt=suffix.suffix,
                    generation_method="dsn_suffix",
                    metadata={
                        "category": category,
                        "strategy": suffix.strategy,
                        "index": idx,
                        "success": True,
                    },
                    artifact=artifact,
                    metrics=metrics,
                    run_metadata=run_meta,
                )
            )

        return outputs
