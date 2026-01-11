"""Adapter for the AART prototype.

This wraps the AART service into the BaseGenerator interface so it can be used
via the /generate-adversarial-prompts endpoint.
"""

import time
import uuid

from services.aart.api import AARTRequest
from services.aart.service import aart_service
from services.prompt_generation.src.generators.base import (
    BaseGenerator,
    GeneratedPrompt,
    GenerationMetrics,
    PromptArtifact,
    RunMetadata,
)


class AARTPromptGenerator(BaseGenerator):
    """Generates region-specific harmful prompts using AART."""

    name = "aart"
    description = "AART: region/task CoT dataset generation"

    def generate(
        self,
        category: str,
        num_prompts: int = 6,
        seed_prompt: str | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate prompts via the AART service.

        The seed prompt is treated as an optional concept override.
        """
        start = time.time()
        run_id = f"aart-{uuid.uuid4().hex[:8]}"

        request = AARTRequest(
            concepts=[seed_prompt] if seed_prompt else None,
            num_prompts=num_prompts,
            # requirements/output_fields captured for future custom schemas
        )
        response = aart_service(request)

        outputs: list[GeneratedPrompt] = []
        for idx, prompt_record in enumerate(response.prompts):
            artifact = PromptArtifact(
                system=None,
                instruction=prompt_record.prompt,
                persona=None,
                suffix=None,
                icl_examples=[],
            )

            metrics = GenerationMetrics(
                latency_ms=int((time.time() - start) * 1000),
            )

            run_meta = RunMetadata(
                method="aart",
                backend=response.metadata.get("backend"),
                run_id=response.metadata.get("run_id", run_id),
                logs_path=None,
                config=response.metadata,
            )

            outputs.append(
                GeneratedPrompt(
                    prompt=prompt_record.prompt,
                    generation_method="aart",
                    metadata={
                        "category": category,
                        "concepts": response.concepts,
                        "region": prompt_record.region,
                        "medium": prompt_record.medium_keyword,
                        "index": idx,
                    },
                    artifact=artifact,
                    metrics=metrics,
                    run_metadata=run_meta,
                )
            )

        return outputs
