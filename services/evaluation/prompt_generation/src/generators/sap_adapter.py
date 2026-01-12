"""Adapter for the SAP (attack-prompt) prototype.

Wraps the SAP service to produce GeneratedPrompt objects.
"""

import time
import uuid

from services.evaluation.attacks.sap.api import SAPRequest
from services.evaluation.attacks.sap.service import sap_service
from services.evaluation.prompt_generation.src.generators.base import (
    BaseGenerator,
    GeneratedPrompt,
    GenerationMetrics,
    PromptArtifact,
    RunMetadata,
)


class SAPPromptGenerator(BaseGenerator):
    """Generates attack prompts using the SAP loop."""

    name = "sap"
    description = "SAP: attack-prompt generate→attack→evaluate loop"

    def generate(
        self,
        category: str,
        num_prompts: int = 3,
        seed_prompt: str | None = None,
    ) -> list[GeneratedPrompt]:
        """Generate SAP-style prompts (offline-friendly by default)."""
        start = time.time()
        run_id = f"sap-{uuid.uuid4().hex[:8]}"

        request = SAPRequest(
            rounds=max(1, num_prompts // 3),
            candidates_per_round=max(1, min(num_prompts, 5)),
            success_threshold=5,
            seeds=None,
        )
        response = sap_service(request)

        outputs: list[GeneratedPrompt] = []
        for _idx, result in enumerate(response.results[:num_prompts]):
            artifact = PromptArtifact(
                system=None,
                instruction=result.prompt,
                persona=None,
                suffix=None,
                icl_examples=[],
            )
            metrics = GenerationMetrics(
                judge_score=result.harmfulness_score,
                latency_ms=int((time.time() - start) * 1000),
            )
            run_meta = RunMetadata(
                method="sap",
                backend=response.metadata.get("attacker_backend"),
                run_id=response.metadata.get("run_id", run_id),
                logs_path=None,
                config=response.metadata,
            )
            outputs.append(
                GeneratedPrompt(
                    prompt=result.prompt,
                    generation_method="sap",
                    metadata={
                        "category": category,
                        "round": result.round,
                        "candidate": result.candidate,
                        "success": result.success,
                        "harmfulness_score": result.harmfulness_score,
                    },
                    artifact=artifact,
                    metrics=metrics,
                    run_metadata=run_meta,
                )
            )

        return outputs
