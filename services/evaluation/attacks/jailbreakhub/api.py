"""API models for JAILBREAKHUB analytics service."""

from pydantic import BaseModel, Field


class JailbreakHubRequest(BaseModel):
    """Request to analyze jailbreak prompts."""

    prompts: list[str] = Field(..., min_length=1, description="Prompts to cluster")
    model_name: str | None = Field(
        None, description="Optional model for ASR replay (if supported)"
    )
    max_samples: int = Field(
        100, ge=1, le=1000, description="Limit prompts analyzed for quick runs"
    )


class JailbreakHubCluster(BaseModel):
    """Cluster assignment for prompts."""

    cluster_id: int
    members: list[str]


class JailbreakHubResponse(BaseModel):
    """JAILBREAKHUB analytics output."""

    method: str
    total_prompts: int
    clusters: list[JailbreakHubCluster]
    metadata: dict
