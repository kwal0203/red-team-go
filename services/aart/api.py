"""API models for the AART service."""

from pydantic import BaseModel, Field


class AARTRequest(BaseModel):
    """Request payload for running AART generation."""

    concepts: list[str] | None = Field(
        None,
        description="Optional list of harmful concepts to target (defaults to built-ins)",
    )
    num_prompts: int = Field(
        6,
        ge=1,
        le=50,
        description="Number of prompts to generate (total across concepts)",
    )
    formats: list[str] | None = Field(
        None, description="Optional mediums/format keywords to include"
    )
    regions: list[str] | None = Field(
        None, description="Optional list of geographic regions to cover"
    )
    backend: str | None = Field(
        None,
        description="LLM backend to use (openai|openrouter); defaults to env configuration",
    )
    max_retries: int = Field(
        2, ge=0, le=5, description="Retries for JSON/schema validation"
    )
    requirements: list[str] | None = Field(
        None,
        description="Additional textual requirements to include in the prompt to the generator",
    )
    output_fields: list[str] | None = Field(
        None,
        description="Optional future schema fields to request (captured for later wiring)",
    )


class AARTPrompt(BaseModel):
    """Single AART prompt record."""

    region: str
    why_prompt_tailored_for_region: str
    medium_keyword: str
    why_prompt_harmful: str
    why_prompt_about_medium: str
    prompt: str


class AARTResponse(BaseModel):
    """Response containing AART prompts and metadata."""

    method: str
    concepts: list[str]
    prompts: list[AARTPrompt]
    metadata: dict
