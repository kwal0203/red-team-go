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


# =============================================================================
# Guardrail Models
# =============================================================================


class GuardrailCheck(BaseModel):
    """Result from a single guardrail check."""

    name: str = Field(..., description="Guardrail identifier (e.g., 'jailbreak')")
    detected: bool = Field(..., description="Whether a violation was detected")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score for the detection"
    )
    explanation: str = Field(..., description="Short description of why flagged")
    category: str | None = Field(None, description="Sub-category if applicable")


class GuardrailEvaluateRequest(BaseModel):
    """Request model for guardrail evaluation (red-team testing)."""

    model: Model
    prompt: str = Field(..., description="The prompt to send to the model and evaluate")
    guardrails: list[str] | None = Field(
        None,
        description="List of guardrails to apply. If None, uses all available.",
        examples=[["jailbreak", "injection", "toxicity", "harmful_content"]],
    )


class GuardrailEvaluateResponse(BaseModel):
    """Response model for guardrail evaluation."""

    prompt: str = Field(..., description="The original prompt")
    model_response: str = Field(..., description="The model's response")
    input_analysis: dict[str, GuardrailCheck] = Field(
        ..., description="Guardrail results for the input prompt"
    )
    output_analysis: dict[str, GuardrailCheck] = Field(
        ..., description="Guardrail results for the model response"
    )
    overall_risk: str = Field(
        ...,
        description="Aggregated risk level",
        examples=["low", "medium", "high", "critical"],
    )
    guardrails_bypassed: list[str] = Field(
        ..., description="List of guardrails that detected violations in output"
    )


class GuardrailProtectRequest(BaseModel):
    """Request model for guardrail protection (production middleware)."""

    input_text: str | None = Field(None, description="Input text to check")
    output_text: str | None = Field(None, description="Output text to check")
    action: str = Field(
        "redact",
        description="Remediation action to take",
        examples=["block", "flag", "redact"],
    )
    guardrails: list[str] | None = Field(
        None,
        description="List of guardrails to apply. If None, uses all available.",
    )


class GuardrailRemediationInfo(BaseModel):
    """Information about remediation action taken."""

    action_taken: str = Field(..., description="The action that was applied")
    explanation: str = Field(..., description="Description of what was done")


class GuardrailProtectResponse(BaseModel):
    """Response model for guardrail protection."""

    allowed: bool = Field(..., description="Whether content should be allowed through")
    input_safe: bool = Field(
        ..., description="Whether input passed checks (if provided)"
    )
    output_safe: bool = Field(
        ..., description="Whether output passed checks (if provided)"
    )
    violations: list[str] = Field(..., description="List of all violations found")
    remediated_output: str | None = Field(
        None, description="Modified output if redaction was applied"
    )
    remediation: GuardrailRemediationInfo | None = Field(
        None, description="Details about remediation action taken"
    )
