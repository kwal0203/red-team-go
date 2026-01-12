"""API models for the SAP (attack-prompt) service."""

from pydantic import BaseModel, Field


class SAPSeedExample(BaseModel):
    """Seed prompt and explanation for the SAP loop."""

    prompt: str = Field(..., description="Attack prompt text")
    explanation: str = Field(..., description="Why this prompt is a good attack")


class SAPRequest(BaseModel):
    """Request payload to run the SAP generate→attack→evaluate loop."""

    rounds: int = Field(3, ge=1, le=10, description="Number of rounds")
    candidates_per_round: int = Field(
        3, ge=1, le=20, description="Candidates per round"
    )
    success_threshold: int = Field(
        5, ge=1, le=10, description="Minimum harmfulness score to keep"
    )
    max_retries: int = Field(2, ge=0, le=5, description="Retries for model calls")
    attacker_backend: str | None = Field(
        None, description="Backend for attacker generation (openai|openrouter)"
    )
    target_backend: str | None = Field(
        None, description="Backend for target model (openai|openrouter)"
    )
    evaluator_backend: str | None = Field(
        None, description="Backend for safety evaluator (openai|openrouter)"
    )
    seeds: list[SAPSeedExample] | None = Field(
        None, description="Optional seed prompts to bootstrap the loop"
    )
    requirements: list[str] | None = Field(
        None,
        description="Additional textual requirements to include in the attack prompt instructions",
    )
    output_fields: list[str] | None = Field(
        None,
        description="Future schema fields to request (captured in metadata for later wiring)",
    )


class SAPResult(BaseModel):
    """Single SAP candidate result."""

    round: int
    candidate: int
    prompt: str
    explanation: str
    target_response: str
    harmfulness_score: int
    success: bool
    duration_s: float


class SAPResponse(BaseModel):
    """Response containing SAP loop results."""

    method: str
    results: list[SAPResult]
    metadata: dict
