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
        description="Backend to use (openai|openrouter|whitebox); defaults to env configuration",
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
    suffix_token_count: int | None = Field(
        None,
        ge=1,
        le=200,
        description="Override whitebox suffix token count (default mirrors experimental code)",
    )
    max_steps: int | None = Field(
        None, ge=1, description="Override optimizer steps for whitebox mode"
    )
    top_k: int | None = Field(
        None, ge=1, description="Override candidate token count for GCG search"
    )
    position_candidates: int | None = Field(
        None, ge=1, description="Override number of gradient positions to explore"
    )
    patience: int | None = Field(
        None, ge=1, description="Override patience before early stopping"
    )
    alpha: float | None = Field(
        None, description="Override refusal unlikelihood weighting"
    )
    seed: int | None = Field(
        None, description="Override RNG seed for whitebox suffix search"
    )
    train_path: str | None = Field(
        None,
        description="Optional training dataset path (JSONL or JSON array with query/target)",
    )
    eval_path: str | None = Field(
        None,
        description="Optional evaluation dataset path (JSONL or JSON array with query/target)",
    )
    max_examples: int | None = Field(
        None, ge=1, description="Limit on training examples loaded"
    )
    eval_max_examples: int | None = Field(
        None, ge=1, description="Limit on evaluation examples loaded"
    )
    model_name_whitebox: str | None = Field(
        None,
        description="Model to use for whitebox optimization (default distilgpt2)",
    )
    refusal_keywords: list[str] | None = Field(
        None, description="Custom refusal keyword list for unlikelihood loss"
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
