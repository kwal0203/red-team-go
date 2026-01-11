"""API models for the DSN (Don't-Say-No) service."""

from pydantic import BaseModel, Field


class DSNRequest(BaseModel):
    """Request payload for DSN suffix generation."""

    seed_prompt: str | None = Field(
        None, description="Optional seed prompt or task description to steer suffixes"
    )
    num_suffixes: int = Field(
        5, ge=1, le=20, description="Number of suffixes to generate"
    )
    max_tokens: int = Field(
        30, ge=5, le=200, description="Approximate max tokens per suffix"
    )
    backend: str | None = Field(
        None,
        description="Backend to use (openai|openrouter); defaults to env configuration",
    )
    strategies: list[str] | None = Field(
        None,
        description="Optional techniques to emphasize (e.g., roleplay, system_override, research_cover)",
    )
    requirements: list[str] | None = Field(
        None,
        description="Additional textual requirements to append to the prompt template",
    )
    output_fields: list[str] | None = Field(
        None,
        description="Future schema fields to request (captured for metadata)",
    )


class DSNSuffix(BaseModel):
    """Single DSN suffix artifact."""

    suffix: str
    strategy: str
    explanation: str


class DSNResponse(BaseModel):
    """Response containing DSN suffixes and metadata."""

    method: str
    seed_prompt: str | None
    suffixes: list[DSNSuffix]
    metadata: dict
