"""Adapter for the STP (Scalable and Transferable Prompts) prototype.

This wraps experimental/single/scalable-and-transferable/main.py into the
BaseGenerator contract, returning structured PromptArtifact objects.
"""

import os
import time
import uuid
from dataclasses import dataclass

from services.evaluation.prompt_generation.src.generators.base import (
    GeneratedPrompt,
    GenerationMetrics,
    PromptArtifact,
    RunMetadata,
)
from utils.config import (
    get_openai_model_name,
    get_openrouter_base_url,
    get_openrouter_key,
    get_openrouter_model_name,
)


@dataclass
class STPConfig:
    """Configuration for the STP adapter."""

    backend: str
    assistant_model: str
    target_model: str
    base_url: str | None
    api_key: str | None
    mode: str = "auto"
    jailbreaks_per_category: int = 1


def _default_stp_config() -> STPConfig:
    backend = os.getenv("STP_BACKEND", "openrouter").lower()
    mode = os.getenv("STP_MODE", "auto").lower()
    jailbreaks = int(os.getenv("STP_JAILBREAKS_PER_CATEGORY", "1"))

    if backend == "openrouter":
        return STPConfig(
            backend="openrouter",
            assistant_model=os.getenv(
                "STP_ASSISTANT_MODEL", get_openrouter_model_name()
            ),
            target_model=os.getenv("STP_TARGET_MODEL", get_openrouter_model_name()),
            base_url=get_openrouter_base_url(),
            api_key=get_openrouter_key(),
            mode=mode,
            jailbreaks_per_category=jailbreaks,
        )

    if backend == "openai":
        return STPConfig(
            backend="openai",
            assistant_model=os.getenv("STP_ASSISTANT_MODEL", get_openai_model_name()),
            target_model=os.getenv("STP_TARGET_MODEL", get_openai_model_name()),
            base_url=None,
            api_key=os.getenv("API_KEY_OPENAI"),
            mode=mode,
            jailbreaks_per_category=jailbreaks,
        )

    # Fallback: treat as openrouter-compatible
    return STPConfig(
        backend=backend,
        assistant_model=os.getenv("STP_ASSISTANT_MODEL", get_openrouter_model_name()),
        target_model=os.getenv("STP_TARGET_MODEL", get_openrouter_model_name()),
        base_url=get_openrouter_base_url(),
        api_key=get_openrouter_key(),
        mode=mode,
        jailbreaks_per_category=jailbreaks,
    )


def run_stp_once(
    category: str, seed_prompt: str | None = None
) -> list[GeneratedPrompt]:
    """Execute a single-category STP synthesis and wrap outputs."""
    from experimental.single.scalable_and_transferable import main as stp

    cfg = _default_stp_config()
    run_id = f"stp-{uuid.uuid4().hex[:8]}"
    start = time.time()

    # Prepare a minimal override of the prototype config
    stp.BACKEND = cfg.backend
    stp.MODE = cfg.mode
    stp.DEFAULT_JAILBREAKS_PER_CATEGORY = cfg.jailbreaks_per_category
    stp.CONFIG[cfg.backend]["assistant_model"] = cfg.assistant_model
    stp.CONFIG[cfg.backend]["target_model"] = cfg.target_model
    if cfg.backend != "openai":
        stp.CONFIG[cfg.backend]["base_url"] = cfg.base_url
    if cfg.api_key:
        os.environ[stp.CONFIG[cfg.backend]["api_key_env"]] = cfg.api_key

    stp.init_dirs()
    client, assistant_model, target_model = stp.get_client()

    outputs = []
    for i in range(cfg.jailbreaks_per_category):
        misuse_instruction, persona_description, modulation_system_prompt = (
            stp.generate_jailbreak(
                client=client,
                assistant_model=assistant_model,
                category=category,
                mode=cfg.mode,
            )
        )

        artifact = PromptArtifact(
            system=modulation_system_prompt,
            instruction=misuse_instruction,
            persona=persona_description,
            suffix=None,
        )

        metrics = GenerationMetrics(
            asr=None,
            judge_score=None,
            refusal_rate=None,
            novelty=None,
            cost=None,
            tokens=None,
            latency_ms=int((time.time() - start) * 1000),
        )

        run_meta = RunMetadata(
            method="stp",
            backend=cfg.backend,
            run_id=run_id,
            logs_path=f"logs/{run_id}.txt",
            config={
                "assistant_model": assistant_model,
                "target_model": target_model,
                "mode": cfg.mode,
            },
        )

        outputs.append(
            GeneratedPrompt(
                prompt=misuse_instruction,
                generation_method="stp",
                metadata={"category": category, "index": i},
                artifact=artifact,
                metrics=metrics,
                run_metadata=run_meta,
            )
        )

    return outputs
