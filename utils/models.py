from typing import Any

from pydantic import BaseModel, Field


class Model(BaseModel):
    """Configuration for a target LLM to evaluate."""

    name: str = Field(
        ..., description="Model identifier (must contain 'openai' or 'huggingface')"
    )
    description: str = Field(..., description="Description of the model")
    base_url: str | None = Field(
        None, description="Base URL for HuggingFace TGI endpoints"
    )


class UserPrompt(BaseModel):
    """Single user prompt for evaluation."""

    prompt: str = Field(..., description="The prompt to evaluate")


class PromptLibrary(BaseModel):
    """Reference to a prompt library file."""

    prompt_library_path: str


# =============================================================================
# Batch Detection Models
# =============================================================================


class DetectionBatchToxicity(BaseModel):
    """Request model for batch toxicity detection."""

    model: Model
    num_samples: int = Field(..., ge=1, description="Number of samples to evaluate")
    random: bool | None = Field(True, description="Randomly sample from database")
    database_prompts: bool | None = Field(True, description="Use prompts from database")
    user_prompts: list[str] | None = Field(
        None, description="User-provided prompts to evaluate"
    )
    user_topics: list[str] | None = Field(
        None, description="Topics to generate prompts for"
    )


class DetectionBatchBias(BaseModel):
    """Request model for batch bias detection."""

    model: Model
    num_samples: int = Field(..., ge=1, description="Number of samples to evaluate")
    random: bool | None = Field(True, description="Randomly sample from database")
    database_prompts: bool | None = Field(True, description="Use prompts from database")
    user_prompts: list[str] | None = Field(
        None, description="User-provided prompts to evaluate"
    )
    user_topics: list[str] | None = Field(
        None, description="Topics to generate prompts for"
    )


class ResultBatch(BaseModel):
    """Response model for batch detection results."""

    result: dict[str, Any]


# =============================================================================
# Realtime Detection Models
# =============================================================================


class DetectionRealtimeToxicity(BaseModel):
    """Request model for realtime toxicity detection of a single prompt."""

    model: Model
    prompt: str = Field(..., description="The prompt to send to the model and evaluate")


class DetectionRealtimeBias(BaseModel):
    """Request model for realtime bias detection of a single prompt."""

    model: Model
    prompt: str = Field(..., description="The prompt to send to the model and evaluate")


class ToxicityScore(BaseModel):
    """Toxicity evaluation scores."""

    openai_score: float = Field(
        ..., ge=0, le=1, description="OpenAI moderator toxicity score"
    )
    paradetox_score: float = Field(
        ..., ge=0, le=1, description="Paradetox model toxicity score"
    )
    average_score: float = Field(..., ge=0, le=1, description="Average of both scores")
    is_toxic: bool = Field(
        ..., description="Whether the response is considered toxic (avg > 0.5)"
    )


class BiasScore(BaseModel):
    """Bias evaluation scores."""

    bias_detected: bool = Field(
        ..., description="Whether bias was detected in the response"
    )
    explanation: str = Field(..., description="Explanation of the bias evaluation")


class ResultRealtimeToxicity(BaseModel):
    """Response model for realtime toxicity detection."""

    prompt: str = Field(..., description="The original prompt")
    model_response: str = Field(..., description="The model's response to the prompt")
    toxicity: ToxicityScore = Field(..., description="Toxicity evaluation results")


class ResultRealtimeBias(BaseModel):
    """Response model for realtime bias detection."""

    prompt: str = Field(..., description="The original prompt")
    model_response: str = Field(..., description="The model's response to the prompt")
    bias: BiasScore = Field(..., description="Bias evaluation results")


# Legacy model for backwards compatibility
class ResultRealtime(BaseModel):
    """Legacy response model for realtime detection."""

    result: str


class ManualDetection(BaseModel):
    """For use in realtime setting (i.e. when chatting)."""

    prompt: UserPrompt
