"""API models for the GPTFUZZER service."""

from pydantic import BaseModel, Field


class GPTFuzzerRequest(BaseModel):
    """Request to run GPTFUZZER template mutation."""

    prompt: str = Field(..., description="Seed prompt or template to mutate")
    num_variants: int = Field(
        10, ge=1, le=100, description="Number of variants to generate"
    )
    max_iterations: int = Field(
        50, ge=1, le=500, description="Iterations for the search loop"
    )


class GPTFuzzerVariant(BaseModel):
    """Single GPTFUZZER variant result."""

    variant: str
    score: float
    lineage: list[str]


class GPTFuzzerResponse(BaseModel):
    """Response containing mutated variants."""

    method: str
    prompt: str
    variants: list[GPTFuzzerVariant]
    metadata: dict
