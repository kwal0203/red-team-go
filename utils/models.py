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


# =============================================================================
# Adversarial Robustness Models (Phase 3, Item 9)
# =============================================================================


class AdversarialRobustnessRequest(BaseModel):
    """Request model for adversarial robustness testing."""

    model: Model
    prompt: str = Field(..., description="The prompt to test with perturbations")
    perturbation_types: list[str] | None = Field(
        None,
        description="Perturbation types to apply",
        examples=[["character", "word", "semantic"]],
    )
    num_variants: int = Field(
        5, ge=1, le=20, description="Number of variants per perturbation type"
    )


class PerturbationVariant(BaseModel):
    """A single perturbed variant and its evaluation."""

    perturbation_type: str = Field(..., description="Type of perturbation applied")
    method: str = Field(..., description="Specific method used")
    perturbed_prompt: str = Field(..., description="The modified prompt")
    model_response: str = Field(..., description="Model's response to the variant")
    blocked: bool = Field(..., description="Whether the model refused")
    bypass_successful: bool = Field(
        ..., description="Whether this variant bypassed safety"
    )
    changes: list[str] = Field(
        default_factory=list, description="Changes made to original"
    )


class AdversarialRobustnessSummary(BaseModel):
    """Summary statistics for robustness testing."""

    total_variants: int = Field(..., description="Total variants tested")
    bypasses: int = Field(..., description="Number of successful bypasses")
    bypass_rate: float = Field(
        ..., ge=0, le=1, description="Rate of successful bypasses"
    )
    most_effective_type: str | None = Field(
        None, description="Most effective perturbation type"
    )


class AdversarialRobustnessResponse(BaseModel):
    """Response model for adversarial robustness testing."""

    original_prompt: str = Field(..., description="The original prompt")
    original_blocked: bool = Field(..., description="Whether original would be blocked")
    variants: list[PerturbationVariant] = Field(..., description="All tested variants")
    summary: AdversarialRobustnessSummary = Field(..., description="Summary statistics")


# =============================================================================
# Stereotype Benchmark Models (Phase 3, Item 10)
# =============================================================================


class StereotypeBenchmarkRequest(BaseModel):
    """Request model for stereotype benchmark evaluation."""

    model: Model
    benchmark: str = Field(
        ...,
        description="Benchmark to use",
        examples=["stereoset", "crows_pairs", "bbq"],
    )
    num_samples: int | None = Field(
        None, ge=1, description="Number of samples to evaluate"
    )
    bias_types: list[str] | None = Field(
        None,
        description="Filter to specific bias types",
        examples=[["gender", "race", "religion"]],
    )
    include_samples: bool = Field(False, description="Include per-sample results")


class BiasTypeMetrics(BaseModel):
    """Metrics for a specific bias type."""

    stereotype_score: float = Field(
        ..., ge=0, le=1, description="Score indicating stereotype preference"
    )
    num_samples: int = Field(..., description="Number of samples evaluated")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional metric details"
    )


class StereotypeBenchmarkResponse(BaseModel):
    """Response model for stereotype benchmark evaluation."""

    benchmark: str = Field(..., description="Benchmark used")
    model: str = Field(..., description="Model evaluated")
    num_samples: int = Field(..., description="Total samples evaluated")
    metrics: dict[str, Any] = Field(..., description="Overall metrics")
    by_bias_type: dict[str, BiasTypeMetrics] = Field(
        ..., description="Metrics by bias type"
    )
    sample_results: list[dict[str, Any]] | None = Field(
        None, description="Per-sample results if requested"
    )


# =============================================================================
# Prompt Generation Models (Phase 3, Item 11)
# =============================================================================


class PromptGenerationRequest(BaseModel):
    """Request model for adversarial prompt generation."""

    model: Model
    target_category: str = Field(
        ...,
        description="Category of prompts to generate",
        examples=["jailbreak", "harmful", "bias", "toxicity"],
    )
    generator: str = Field(
        "llm",
        description="Generator method to use",
        examples=["llm", "genetic", "pair"],
    )
    num_prompts: int = Field(
        10, ge=1, le=50, description="Number of prompts to generate"
    )
    seed_prompt: str | None = Field(
        None, description="Optional seed prompt for generation"
    )
    evaluate: bool = Field(
        True, description="Whether to evaluate prompts against target model"
    )


class GeneratedPromptResult(BaseModel):
    """A generated prompt with optional evaluation."""

    prompt: str = Field(..., description="The generated prompt")
    generation_method: str = Field(..., description="Method used to generate")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata"
    )
    evaluation: dict[str, Any] | None = Field(
        None, description="Evaluation results if evaluated"
    )


class PromptGenerationSummary(BaseModel):
    """Summary statistics for prompt generation."""

    total_generated: int = Field(..., description="Total prompts generated")
    successful_bypasses: int = Field(0, description="Number of successful bypasses")
    bypass_rate: float = Field(0, ge=0, le=1, description="Rate of successful bypasses")


class PromptGenerationResponse(BaseModel):
    """Response model for prompt generation."""

    generator: str = Field(..., description="Generator method used")
    target_category: str = Field(..., description="Target category")
    prompts: list[GeneratedPromptResult] = Field(
        ..., description="Generated prompts with evaluations"
    )
    summary: PromptGenerationSummary = Field(..., description="Summary statistics")


# =============================================================================
# Consistency & Reliability Models (Phase 4)
# =============================================================================


class ConsistencyReliabilityRequest(BaseModel):
    """Request model for consistency and reliability testing."""

    model: Model
    prompt: str = Field(
        ..., description="The prompt to test for consistency/reliability"
    )
    test_types: list[str] | None = Field(
        None,
        description="Test types to run (default: all)",
        examples=[
            ["sycophancy", "stability", "self_consistency", "instruction_following"]
        ],
    )
    num_samples: int = Field(
        5, ge=1, le=20, description="Number of samples/variations per test"
    )
    sycophancy_topics: list[str] | None = Field(
        None,
        description="Topics for sycophancy testing (default: auto-generated)",
    )
    instruction_constraints: list[str] | None = Field(
        None,
        description="Custom constraints for instruction following test",
    )


class TestResult(BaseModel):
    """Result from a single test execution."""

    test_type: str = Field(..., description="Type of test run")
    score: float = Field(
        ..., ge=0, le=1, description="Score from 0-1 (higher = better)"
    )
    passed: bool = Field(..., description="Whether the test passed threshold")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Test-specific details"
    )
    samples: list[dict[str, Any]] | None = Field(
        None, description="Per-sample results if available"
    )


class ConsistencyReliabilitySummary(BaseModel):
    """Summary statistics for all tests."""

    tests_run: int = Field(..., description="Number of tests executed")
    tests_passed: int = Field(..., description="Number of tests that passed")
    overall_score: float = Field(
        ..., ge=0, le=1, description="Weighted average of all test scores"
    )
    reliability_grade: str = Field(
        ...,
        description="Overall reliability grade",
        examples=["A", "B", "C", "D", "F"],
    )


class ConsistencyReliabilityResponse(BaseModel):
    """Response model for consistency and reliability testing."""

    model: str = Field(..., description="Model tested")
    prompt: str = Field(..., description="Original prompt tested")
    results: dict[str, TestResult] = Field(..., description="Results by test type")
    summary: ConsistencyReliabilitySummary = Field(
        ..., description="Summary statistics"
    )


# =============================================================================
# Misinformation & Factuality Models (Phase 4)
# =============================================================================


class MisinformationFactualityRequest(BaseModel):
    """Request model for misinformation and factuality testing."""

    model: Model
    prompt: str = Field(
        ..., description="The base topic/context for factuality testing"
    )
    test_types: list[str] | None = Field(
        None,
        description="Test types to run (default: all)",
        examples=[
            [
                "knowledge_cutoff",
                "temporal_reasoning",
                "confidence_calibration",
                "citation_verification",
            ]
        ],
    )
    num_samples: int = Field(
        5, ge=1, le=20, description="Number of questions/samples per test"
    )
    knowledge_cutoff_date: str | None = Field(
        None,
        description="Expected knowledge cutoff date for verification (e.g., '2024-01')",
    )
    temporal_questions: list[str] | None = Field(
        None,
        description="Custom temporal reasoning questions",
    )


class FactualityTestResult(BaseModel):
    """Result from a single factuality test execution."""

    test_type: str = Field(..., description="Type of test run")
    score: float = Field(
        ..., ge=0, le=1, description="Score from 0-1 (higher = better)"
    )
    passed: bool = Field(..., description="Whether the test passed threshold")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Test-specific details"
    )
    samples: list[dict[str, Any]] | None = Field(
        None, description="Per-sample results if available"
    )


class MisinformationFactualitySummary(BaseModel):
    """Summary statistics for all factuality tests."""

    tests_run: int = Field(..., description="Number of tests executed")
    tests_passed: int = Field(..., description="Number of tests that passed")
    overall_score: float = Field(
        ..., ge=0, le=1, description="Weighted average of all test scores"
    )
    factuality_grade: str = Field(
        ...,
        description="Overall factuality grade",
        examples=["A", "B", "C", "D", "F"],
    )


class MisinformationFactualityResponse(BaseModel):
    """Response model for misinformation and factuality testing."""

    model: str = Field(..., description="Model tested")
    prompt: str = Field(..., description="Original topic/context tested")
    results: dict[str, FactualityTestResult] = Field(
        ..., description="Results by test type"
    )
    summary: MisinformationFactualitySummary = Field(
        ..., description="Summary statistics"
    )
