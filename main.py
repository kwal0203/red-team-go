"""RedTeamGo - LLM Red Teaming and Safety Evaluation API."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded

from services.adversarial_robustness.service import adversarial_robustness_service
from services.bias_detection_dbias.service import (
    bias_detection_realtime_service,
    dbias_service,
)
from services.consistency_reliability.service import consistency_reliability_service
from services.datasets import (
    DatasetCategory,
    DatasetNotFoundError,
    get_dataset,
    get_dataset_info,
    list_datasets,
)
from services.gptfuzzer.api import GPTFuzzerRequest, GPTFuzzerResponse
from services.gptfuzzer.service import gptfuzzer_service
from services.guardrails.service import (
    guardrails_evaluate_service,
    guardrails_protect_service,
)
from services.jailbreakhub.api import JailbreakHubRequest, JailbreakHubResponse
from services.jailbreakhub.service import jailbreakhub_service
from services.misinformation_factuality.service import (
    misinformation_factuality_service,
)
from services.privacy_redteam.service import privacy_redteam_service
from services.prompt_generation.service import prompt_generation_service
from services.refusal_consistency.service import refusal_consistency_service
from services.stereotype_benchmarks.service import stereotype_benchmark_service
from services.toxicity_detection.service import (
    toxicity_detection_realtime_service,
    toxicity_detection_service,
)
from utils.artifact_storage import store_evaluation_artifact
from utils.auth import APIKeyDep
from utils.models import (
    AdversarialRobustnessRequest,
    AdversarialRobustnessResponse,
    AuthErrorResponse,
    ConsistencyReliabilityRequest,
    ConsistencyReliabilityResponse,
    DatasetInfoResponse,
    DatasetListResponse,
    DatasetSampleRequest,
    DatasetSampleResponse,
    DetectionBatchBias,
    DetectionBatchToxicity,
    DetectionRealtimeBias,
    DetectionRealtimeToxicity,
    ErrorResponse,
    GuardrailEvaluateRequest,
    GuardrailEvaluateResponse,
    GuardrailProtectRequest,
    GuardrailProtectResponse,
    MisinformationFactualityRequest,
    MisinformationFactualityResponse,
    ModelConfidenceRequest,
    ModelConfidenceResponse,
    PrivacyRedTeamRequest,
    PrivacyRedTeamResponse,
    PromptGenerationRequest,
    PromptGenerationResponse,
    RateLimitErrorResponse,
    RefusalConsistencyRequest,
    RefusalConsistencyResponse,
    ResultBatch,
    ResultRealtimeBias,
    ResultRealtimeToxicity,
    StereotypeBenchmarkRequest,
    StereotypeBenchmarkResponse,
    ValidationErrorResponse,
)
from utils.rate_limit import (
    RATE_LIMIT_BATCH,
    RATE_LIMIT_HEALTH,
    RATE_LIMIT_REALTIME,
    limiter,
    rate_limit_exceeded_handler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Common error responses for authenticated endpoints
COMMON_RESPONSES = {
    401: {"model": AuthErrorResponse, "description": "Authentication failed"},
    422: {"model": ValidationErrorResponse, "description": "Validation error"},
    429: {"model": RateLimitErrorResponse, "description": "Rate limit exceeded"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}

# API Tags for endpoint grouping
TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Health check and status endpoints.",
    },
    {
        "name": "Toxicity Detection",
        "description": "Evaluate LLM outputs for toxic content using ensemble moderator models.",
    },
    {
        "name": "Bias Detection",
        "description": "Detect bias in LLM responses using self-evaluation methodology.",
    },
    {
        "name": "Guardrails",
        "description": "Input/output safety guardrails for jailbreak, injection, and harmful content detection.",
    },
    {
        "name": "Adversarial Testing",
        "description": "Test model robustness against adversarial perturbations and attacks.",
    },
    {
        "name": "Stereotype Benchmarks",
        "description": "Evaluate models using established stereotype and fairness benchmarks.",
    },
    {
        "name": "Prompt Generation",
        "description": "Generate adversarial prompts using LLM, Genetic Algorithm, or PAIR methods.",
    },
    {
        "name": "Consistency & Reliability",
        "description": "Test model consistency, stability, and instruction following.",
    },
    {
        "name": "Misinformation & Factuality",
        "description": "Evaluate factual accuracy, temporal reasoning, and confidence calibration.",
    },
    {
        "name": "Refusal Consistency",
        "description": "Test if models consistently refuse harmful requests under adversarial conditions.",
    },
    {
        "name": "Privacy Red Team",
        "description": "Active probing for training data extraction, membership inference, and prompt leakage.",
    },
    {
        "name": "Hallucination Detection",
        "description": "Detect hallucinations using model confidence from token log probabilities.",
    },
    {
        "name": "Datasets",
        "description": "Access and manage red-teaming benchmark datasets (StereoSet, CrowS-Pairs, BBQ, etc.).",
    },
    {
        "name": "Adversarial Search",
        "description": "Iterative jailbreak search and mutation methods (e.g., GPTFUZZER).",
    },
    {
        "name": "Analytics",
        "description": "Jailbreak clustering and coverage analytics (JAILBREAKHUB).",
    },
]

# Create FastAPI app
app = FastAPI(
    title="RedTeamGo API",
    description="""
## LLM Red Teaming and Safety Evaluation Platform

RedTeamGo provides comprehensive tools for evaluating Large Language Model (LLM) safety,
including toxicity detection, bias analysis, adversarial robustness testing, and more.

### Features

- **Toxicity & Bias Detection**: Batch and realtime evaluation using ensemble models
- **Safety Guardrails**: Input/output filtering for jailbreaks, prompt injection, harmful content
- **Adversarial Testing**: Character, word, and semantic perturbations to test robustness
- **Stereotype Benchmarks**: StereoSet, CrowS-Pairs, BBQ evaluation
- **Prompt Generation**: LLM-based, Genetic Algorithm, and PAIR adversarial prompt generation
- **Consistency Testing**: Sycophancy, stability, self-consistency, instruction following
- **Factuality Testing**: Knowledge cutoff, temporal reasoning, confidence calibration
- **Refusal Testing**: Paraphrase, pressure, multi-turn, context switching attacks
- **Privacy Red Team**: Training data extraction, membership inference, prompt leakage
- **Hallucination Detection**: Model confidence analysis using token log probabilities

### Authentication

All endpoints (except health checks) require an API key via the `X-API-Key` header.

### Rate Limits

- **Batch endpoints**: 10 requests/minute
- **Realtime endpoints**: 30 requests/minute
- **Health endpoints**: 60 requests/minute
""",
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "RedTeamGo Support",
        "url": "https://github.com/kwal0203/red-team-go",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add rate limiter state to app
app.state.limiter = limiter

# Register rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Add CORS middleware
# TODO: Configure allowed origins based on environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Prometheus instrumentation
Instrumentator().instrument(app).expose(app)


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get("/health", tags=["Health"])
@limiter.limit(RATE_LIMIT_HEALTH)
async def health_check(request: Request):
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/", tags=["Health"])
@limiter.limit(RATE_LIMIT_HEALTH)
def read_root(request: Request):
    """Root endpoint returning service status."""
    return {"service": "online"}


# =============================================================================
# Batch Detection Endpoints
# =============================================================================


@app.post(
    "/toxicity-detection-batch",
    response_model=ResultBatch,
    tags=["Toxicity Detection"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid model configuration"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def toxicity_detection_batch(
    request: Request,
    args: DetectionBatchToxicity,
    api_key: APIKeyDep,
):
    """
    Batch toxicity detection endpoint.

    Evaluates multiple prompts for toxicity using an ensemble of moderator models
    (OpenAI Moderator + Paradetox).

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Batch detection configuration including model and prompts.

    Returns:
        ResultBatch containing toxicity evaluation results for all samples.
    """
    logger.info(f"Toxicity batch detection request: {len(args.user_prompts)} prompts")
    try:
        toxicity_result = toxicity_detection_service(**args.model_dump())
        response = ResultBatch(result=toxicity_result)
        store_evaluation_artifact(
            request,
            "toxicity_batch",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Toxicity batch detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e


@app.post(
    "/bias-detection-batch",
    response_model=ResultBatch,
    tags=["Bias Detection"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid model configuration"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def bias_detection_batch(
    request: Request,
    args: DetectionBatchBias,
    api_key: APIKeyDep,
):
    """
    Batch bias detection endpoint.

    Evaluates multiple prompts for bias using self-evaluation methodology.

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Batch detection configuration including model and prompts.

    Returns:
        ResultBatch containing bias evaluation results for all samples.
    """
    logger.info(f"Bias batch detection request: {len(args.user_prompts)} prompts")
    try:
        dbias_result = dbias_service(**args.model_dump())
        response = ResultBatch(result=dbias_result)
        store_evaluation_artifact(
            request,
            "bias_batch",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Bias batch detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e


# =============================================================================
# Realtime Detection Endpoints
# =============================================================================


@app.post(
    "/toxicity-detection-realtime",
    response_model=ResultRealtimeToxicity,
    tags=["Toxicity Detection"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid model configuration"},
    },
)
@limiter.limit(RATE_LIMIT_REALTIME)
def toxicity_detection_realtime(
    request: Request,
    args: DetectionRealtimeToxicity,
    api_key: APIKeyDep,
):
    """
    Realtime toxicity detection endpoint.

    Sends a single prompt to the target model, evaluates the response for toxicity,
    and returns both the response and toxicity scores.

    Requires: X-API-Key header
    Rate Limit: 30 requests/minute

    Args:
        args: Realtime detection request with model config and prompt.

    Returns:
        ResultRealtimeToxicity containing the prompt, model response, and toxicity scores.
    """
    logger.info(f"Toxicity realtime detection request for model: {args.model.name}")
    try:
        result = toxicity_detection_realtime_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
        )
        response = ResultRealtimeToxicity(**result)
        store_evaluation_artifact(
            request,
            "toxicity_realtime",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Toxicity detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e


@app.post(
    "/bias-detection-realtime",
    response_model=ResultRealtimeBias,
    tags=["Bias Detection"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid model configuration"},
    },
)
@limiter.limit(RATE_LIMIT_REALTIME)
def bias_detection_realtime(
    request: Request,
    args: DetectionRealtimeBias,
    api_key: APIKeyDep,
):
    """
    Realtime bias detection endpoint.

    Sends a single prompt to the target model, evaluates the response for bias,
    and returns both the response and bias assessment.

    Requires: X-API-Key header
    Rate Limit: 30 requests/minute

    Args:
        args: Realtime detection request with model config and prompt.

    Returns:
        ResultRealtimeBias containing the prompt, model response, and bias assessment.
    """
    logger.info(f"Bias realtime detection request for model: {args.model.name}")
    try:
        result = bias_detection_realtime_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
        )
        response = ResultRealtimeBias(**result)
        store_evaluation_artifact(
            request,
            "bias_realtime",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e


# =============================================================================
# Guardrail Endpoints
# =============================================================================


@app.post(
    "/evaluate/guardrails",
    response_model=GuardrailEvaluateResponse,
    tags=["Guardrails"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid model configuration"},
    },
)
@limiter.limit(RATE_LIMIT_REALTIME)
def evaluate_guardrails(
    request: Request,
    args: GuardrailEvaluateRequest,
    api_key: APIKeyDep,
):
    """
    Evaluate guardrails for red-team testing.

    Sends a prompt to the target model and evaluates both the input prompt
    and the model's response for various safety violations. Use this to test
    whether a model's guardrails can be bypassed.

    Requires: X-API-Key header
    Rate Limit: 30 requests/minute

    Args:
        args: Evaluation request with model config, prompt, and optional guardrail filter.

    Returns:
        GuardrailEvaluateResponse with input/output analysis and bypassed guardrails.
    """
    logger.info(f"Guardrail evaluation request for model: {args.model.name}")
    try:
        result = guardrails_evaluate_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
            guardrails=args.guardrails,
        )
        response = GuardrailEvaluateResponse(**result)
        store_evaluation_artifact(
            request,
            "guardrails_evaluate",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Guardrail evaluation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Evaluation failed: {str(e)}"
        ) from e


@app.post(
    "/protect/guardrails",
    response_model=GuardrailProtectResponse,
    tags=["Guardrails"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
@limiter.limit(RATE_LIMIT_REALTIME)
def protect_guardrails(
    request: Request,
    args: GuardrailProtectRequest,
    api_key: APIKeyDep,
):
    """
    Protect mode guardrail check for production middleware.

    Checks input and/or output text for safety violations and applies
    the specified remediation action (block, flag, or redact).

    Requires: X-API-Key header
    Rate Limit: 30 requests/minute

    Args:
        args: Protect request with input/output text and action to take.

    Returns:
        GuardrailProtectResponse with safety status and any remediated content.
    """
    logger.info(f"Guardrail protect request with action: {args.action}")
    try:
        result = guardrails_protect_service(
            input_text=args.input_text,
            output_text=args.output_text,
            action=args.action,
            guardrails=args.guardrails,
        )
        response = GuardrailProtectResponse(**result)
        store_evaluation_artifact(
            request,
            "guardrails_protect",
            args,
            response,
            api_key=api_key,
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Guardrail protect check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Protection check failed: {str(e)}"
        ) from e


# =============================================================================
# Adversarial Robustness Endpoints (Phase 3, Item 9)
# =============================================================================


@app.post(
    "/adversarial-robustness",
    response_model=AdversarialRobustnessResponse,
    tags=["Adversarial Testing"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def adversarial_robustness(
    request: Request,
    args: AdversarialRobustnessRequest,
    api_key: APIKeyDep,
):
    """
    Test adversarial robustness of model guardrails.

    Applies various text perturbations (character-level, word-level, semantic)
    to test if safety guardrails can be bypassed.

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Robustness test request with model config, prompt, and perturbation settings.

    Returns:
        AdversarialRobustnessResponse with all variants tested and bypass statistics.
    """
    logger.info(
        f"Adversarial robustness test for model: {args.model.name}, "
        f"perturbations: {args.perturbation_types}"
    )
    try:
        result = adversarial_robustness_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
            perturbation_types=args.perturbation_types,
            num_variants=args.num_variants,
        )
        response = AdversarialRobustnessResponse(**result)
        store_evaluation_artifact(
            request,
            "adversarial_robustness",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Adversarial robustness test failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Robustness test failed: {str(e)}"
        ) from e


# =============================================================================
# Stereotype Benchmark Endpoints (Phase 3, Item 10)
# =============================================================================


@app.post(
    "/stereotype-benchmark",
    response_model=StereotypeBenchmarkResponse,
    tags=["Stereotype Benchmarks"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
        501: {"model": ErrorResponse, "description": "Dataset not loaded"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def stereotype_benchmark(
    request: Request,
    args: StereotypeBenchmarkRequest,
    api_key: APIKeyDep,
):
    """
    Evaluate model for stereotypical biases using established benchmarks.

    Supports StereoSet, CrowS-Pairs, and BBQ benchmarks.
    Note: Full datasets must be loaded separately. Uses sample data by default.

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Benchmark request with model config, benchmark name, and optional filters.

    Returns:
        StereotypeBenchmarkResponse with stereotype scores and per-bias-type metrics.
    """
    logger.info(
        f"Stereotype benchmark for model: {args.model.name}, "
        f"benchmark: {args.benchmark}"
    )
    try:
        result = stereotype_benchmark_service(
            model=args.model.model_dump(),
            benchmark=args.benchmark,
            num_samples=args.num_samples,
            bias_types=args.bias_types,
            include_samples=args.include_samples,
        )
        response = StereotypeBenchmarkResponse(**result)
        store_evaluation_artifact(
            request,
            "stereotype_benchmark",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name, "benchmark": args.benchmark},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        logger.warning(f"Dataset not loaded: {e}")
        raise HTTPException(
            status_code=501, detail=f"Dataset not loaded: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Stereotype benchmark failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Benchmark failed: {str(e)}"
        ) from e


# =============================================================================
# Prompt Generation Endpoints (Phase 3, Item 11)
# =============================================================================


@app.post(
    "/generate-adversarial-prompts",
    response_model=PromptGenerationResponse,
    tags=["Prompt Generation"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
        501: {"model": ErrorResponse, "description": "Generator not implemented"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def generate_adversarial_prompts(
    request: Request,
    args: PromptGenerationRequest,
    api_key: APIKeyDep,
):
    """
    Generate adversarial prompts for red-teaming LLMs.

    Uses the specified generator method to create adversarial prompts
    and optionally evaluates them against the target model.

    Generators:
    - "llm": LLM-based prompt generation using category-specific templates
    - "genetic": Genetic algorithm with mutation, crossover, and fitness evaluation
    - "pair": PAIR (Prompt Automatic Iterative Refinement) method

    Categories: jailbreak, harmful, bias, toxicity

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Generation request with model config, category, generator, and settings.

    Returns:
        PromptGenerationResponse with generated prompts and bypass statistics.
    """
    logger.info(
        f"Prompt generation for model: {args.model.name}, "
        f"category: {args.target_category}, generator: {args.generator}"
    )
    try:
        result = prompt_generation_service(
            model=args.model.model_dump(),
            target_category=args.target_category,
            generator=args.generator,
            num_prompts=args.num_prompts,
            seed_prompt=args.seed_prompt,
            evaluate=args.evaluate,
        )
        response = PromptGenerationResponse(**result)
        store_evaluation_artifact(
            request,
            "prompt_generation",
            args,
            response,
            api_key=api_key,
            extra_metadata={
                "model": args.model.name,
                "generator": args.generator,
                "target_category": args.target_category,
            },
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        logger.warning(f"Generator not implemented: {e}")
        raise HTTPException(
            status_code=501, detail=f"Generator not implemented: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {str(e)}"
        ) from e


# =============================================================================
# Consistency & Reliability Endpoints (Phase 4)
# =============================================================================


@app.post(
    "/consistency-reliability",
    response_model=ConsistencyReliabilityResponse,
    tags=["Consistency & Reliability"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def consistency_reliability(
    request: Request,
    args: ConsistencyReliabilityRequest,
    api_key: APIKeyDep,
):
    """
    Test model consistency and reliability.

    Runs a suite of tests to evaluate how consistent and reliable
    a model's responses are under various conditions.

    Test Types:
    - "sycophancy": Tests if model changes opinions when challenged
    - "stability": Tests response consistency under prompt paraphrasing
    - "self_consistency": Tests consistency across multiple generations
    - "instruction_following": Tests adherence to formatting constraints

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Test request with model config, prompt, and test options.

    Returns:
        ConsistencyReliabilityResponse with per-test results and overall grade.
    """
    logger.info(
        f"Consistency/reliability test for model: {args.model.name}, "
        f"tests: {args.test_types or 'all'}"
    )
    try:
        result = consistency_reliability_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
            test_types=args.test_types,
            num_samples=args.num_samples,
            sycophancy_topics=args.sycophancy_topics,
            instruction_constraints=args.instruction_constraints,
        )
        response = ConsistencyReliabilityResponse(**result)
        store_evaluation_artifact(
            request,
            "consistency_reliability",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Consistency/reliability test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}") from e


# =============================================================================
# Misinformation & Factuality Endpoints (Phase 4)
# =============================================================================


@app.post(
    "/misinformation-factuality",
    response_model=MisinformationFactualityResponse,
    tags=["Misinformation & Factuality"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def misinformation_factuality(
    request: Request,
    args: MisinformationFactualityRequest,
    api_key: APIKeyDep,
):
    """
    Test model for misinformation and factuality issues.

    Runs a suite of tests to evaluate how factual and truthful
    a model's responses are.

    Test Types:
    - "knowledge_cutoff": Tests if model acknowledges its training date limits
    - "temporal_reasoning": Tests date/time arithmetic and ordering
    - "confidence_calibration": Tests if confidence matches accuracy
    - "citation_verification": Tests self-consistency of cited claims

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Test request with model config, prompt, and test options.

    Returns:
        MisinformationFactualityResponse with per-test results and overall grade.
    """
    try:
        result = misinformation_factuality_service(
            model=args.model.model_dump(),
            prompt=args.prompt,
            test_types=args.test_types,
            num_samples=args.num_samples,
            knowledge_cutoff_date=args.knowledge_cutoff_date,
            temporal_questions=args.temporal_questions,
        )
        response = MisinformationFactualityResponse(**result)
        store_evaluation_artifact(
            request,
            "misinformation_factuality",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Misinformation/factuality test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}") from e


# =============================================================================
# Refusal Consistency Endpoints (Item 5.3)
# =============================================================================


@app.post(
    "/refusal-consistency",
    response_model=RefusalConsistencyResponse,
    tags=["Refusal Consistency"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def refusal_consistency(
    request: Request,
    args: RefusalConsistencyRequest,
    api_key: APIKeyDep,
):
    """
    Test model refusal consistency under adversarial conditions.

    Tests if a model consistently refuses harmful requests under various
    adversarial conditions including paraphrasing, user pressure,
    multi-turn escalation, and context switching.

    Test Types:
    - "paraphrase": Tests refusal consistency across paraphrased requests
    - "pressure": Tests refusal under user manipulation/pressure
    - "multi_turn": Tests refusal under gradual trust-building escalation
    - "context_switching": Tests refusal when harmful content is in benign context

    Harm Categories:
    - "weapons", "illegal_activities", "self_harm", "violence", "cbrn", "dual_use"

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Test request with model config and test options.

    Returns:
        RefusalConsistencyResponse with per-test results and overall grade.
    """
    logger.info(f"Refusal consistency test for model: {args.model.name}")
    try:
        result = refusal_consistency_service(
            model=args.model.model_dump(),
            test_types=args.test_types,
            harm_categories=args.harm_categories,
            num_samples=args.num_samples,
            custom_harmful_prompts=args.custom_harmful_prompts,
        )
        response = RefusalConsistencyResponse(**result)
        store_evaluation_artifact(
            request,
            "refusal_consistency",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Refusal consistency test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}") from e


# =============================================================================
# Privacy Red Team Endpoints (Item 5.5)
# =============================================================================


@app.post(
    "/privacy-redteam",
    response_model=PrivacyRedTeamResponse,
    tags=["Privacy Red Team"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def privacy_redteam(
    request: Request,
    args: PrivacyRedTeamRequest,
    api_key: APIKeyDep,
):
    """
    Test model for privacy vulnerabilities through active red teaming.

    Performs active probing tests to detect:
    - Training data extraction: Tests if model leaks memorized data
    - Membership inference: Tests if model reveals training data presence
    - Prompt/system leakage: Tests if model reveals confidential instructions

    Test Types:
    - "training_extraction": Probes for memorized training data
    - "membership_inference": Tests knowledge boundary and confidence
    - "prompt_leakage": Attempts to extract system prompts

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Test request with model config, test options, and optional system prompt.

    Returns:
        PrivacyRedTeamResponse with per-test results and privacy grade.
    """
    logger.info(f"Privacy red team test for model: {args.model.name}")
    try:
        result = privacy_redteam_service(
            model=args.model.model_dump(),
            test_types=args.test_types,
            num_samples=args.num_samples,
            system_prompt=args.system_prompt,
            custom_probes=args.custom_probes,
        )
        response = PrivacyRedTeamResponse(**result)
        store_evaluation_artifact(
            request,
            "privacy_redteam",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Privacy red team test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}") from e


# =============================================================================
# Hallucination Detection Endpoints
# =============================================================================


@app.post(
    "/hallucination-confidence",
    response_model=ModelConfidenceResponse,
    tags=["Hallucination Detection"],
    responses={
        **COMMON_RESPONSES,
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def hallucination_confidence(
    request: Request,
    args: ModelConfidenceRequest,
    api_key: APIKeyDep,
):
    """
    Evaluate hallucination risk using model confidence.

    Analyzes token-level log probabilities to estimate the likelihood
    of hallucination in model outputs. Lower confidence suggests higher
    risk of hallucination.

    Based on: "Looking for a Needle in a Haystack: A Comprehensive Study of
    Hallucinations in Neural Machine Translation" (Guerreiro et al., 2023)

    Confidence Methods:
    - "geometric": Sequence probability (default, most robust)
    - "average": Mean token probability
    - "minimum": Worst-case token confidence (pessimistic)
    - "entropy": Information-theoretic uncertainty
    - "variance": Consistency of confidence across tokens

    Risk Levels:
    - "low": Score >= 70 (unlikely hallucination)
    - "medium": Score 50-69 (some uncertainty)
    - "high": Score 30-49 (potential hallucination)
    - "critical": Score < 30 (likely hallucination)

    Requires: X-API-Key header
    Rate Limit: 10 requests/minute

    Args:
        args: Request with model config, prompt, and confidence method.

    Returns:
        ModelConfidenceResponse with confidence score, risk level, and details.
    """
    from services.hallucination_detection_model_confidence import evaluate_confidence

    logger.info(f"Hallucination confidence check for model: {args.model.name}")
    try:
        result = evaluate_confidence(
            prompt=args.prompt,
            model_name=args.model.model_name or "gpt-3.5-turbo",
            method=args.method,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            include_all_methods=args.include_all_methods,
        )
        response = ModelConfidenceResponse(**result.to_dict())
        store_evaluation_artifact(
            request,
            "model_confidence",
            args,
            response,
            api_key=api_key,
            extra_metadata={"model": args.model.name},
        )
        return response
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Hallucination confidence check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}") from e


# =============================================================================
# Dataset Endpoints
# =============================================================================


@app.get(
    "/datasets",
    response_model=DatasetListResponse,
    tags=["Datasets"],
)
@limiter.limit(RATE_LIMIT_HEALTH)
def list_available_datasets(
    request: Request,
    category: str | None = None,
):
    """
    List all available datasets.

    Returns metadata about all registered datasets, optionally filtered by category.
    This endpoint is public and does not require authentication.

    Categories:
    - "stereotype": Bias and stereotype benchmarks (StereoSet, CrowS-Pairs, BBQ)
    - "jailbreak": Jailbreak prompt datasets
    - "toxicity": Toxicity evaluation datasets
    - "harmful": Harmful content datasets
    - "bias": General bias datasets

    Rate Limit: 60 requests/minute

    Args:
        category: Optional category filter.

    Returns:
        DatasetListResponse with list of available datasets.
    """
    logger.info(f"Listing datasets with category filter: {category}")
    try:
        # Convert category string to enum if provided
        cat_enum = None
        if category:
            try:
                cat_enum = DatasetCategory(category)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category '{category}'. Valid categories: {[c.value for c in DatasetCategory]}",
                ) from e

        datasets = list_datasets(category=cat_enum)
        return DatasetListResponse(
            datasets=[
                DatasetInfoResponse(
                    name=d.name,
                    category=d.category.value,
                    description=d.description,
                    source=d.source,
                    citation=d.citation,
                    size=d.size,
                    huggingface_id=d.huggingface_id,
                )
                for d in datasets
            ],
            total=len(datasets),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list datasets: {str(e)}"
        ) from e


@app.get(
    "/datasets/{name}",
    response_model=DatasetInfoResponse,
    tags=["Datasets"],
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
@limiter.limit(RATE_LIMIT_HEALTH)
def get_dataset_details(
    request: Request,
    name: str,
):
    """
    Get details about a specific dataset.

    Returns metadata about the specified dataset including its source,
    citation, size, and HuggingFace ID if available.
    This endpoint is public and does not require authentication.

    Rate Limit: 60 requests/minute

    Args:
        name: The dataset name (e.g., "stereoset", "crows_pairs", "bbq").

    Returns:
        DatasetInfoResponse with dataset metadata.
    """
    logger.info(f"Getting info for dataset: {name}")
    try:
        info = get_dataset_info(name)
        return DatasetInfoResponse(
            name=info.name,
            category=info.category.value,
            description=info.description,
            source=info.source,
            citation=info.citation,
            size=info.size,
            huggingface_id=info.huggingface_id,
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset info: {str(e)}"
        ) from e


@app.post(
    "/datasets/{name}/sample",
    response_model=DatasetSampleResponse,
    tags=["Datasets"],
    responses={
        **COMMON_RESPONSES,
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
@limiter.limit(RATE_LIMIT_REALTIME)
def get_dataset_samples(
    request: Request,
    name: str,
    args: DatasetSampleRequest,
    api_key: APIKeyDep,
):
    """
    Get sample items from a dataset.

    Returns sample items from the specified dataset without loading the full dataset.
    Useful for testing and demos.

    Requires: X-API-Key header
    Rate Limit: 30 requests/minute

    Args:
        name: The dataset name (e.g., "stereoset", "crows_pairs", "bbq").
        args: Request with number of samples to retrieve.

    Returns:
        DatasetSampleResponse with sample items.
    """
    logger.info(f"Getting {args.num_samples} samples from dataset: {name}")
    try:
        loader = get_dataset(name)
        samples = loader.get_sample(num_samples=args.num_samples)

        return DatasetSampleResponse(
            dataset_name=name,
            samples=[s.to_dict() for s in samples],
            num_samples=len(samples),
            total_available=loader.info.size,
        )
    except DatasetNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get dataset samples: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get samples: {str(e)}"
        ) from e


# =============================================================================
# Adversarial Search Endpoints
# =============================================================================


@app.post(
    "/gptfuzzer",
    response_model=GPTFuzzerResponse,
    tags=["Adversarial Search"],
    responses={
        **COMMON_RESPONSES,
        200: {"description": "GPTFUZZER variant generation successful"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def run_gptfuzzer(
    request: Request,
    args: GPTFuzzerRequest,
    api_key: APIKeyDep,
):
    """Run GPTFUZZER-style template mutation."""
    logger.info(
        f"GPTFUZZER run: prompt length={len(args.prompt)}, "
        f"variants={args.num_variants}, iterations={args.max_iterations}"
    )
    try:
        result = gptfuzzer_service(
            model={"name": "default"},
            prompt=args.prompt,
            num_variants=args.num_variants,
            max_iterations=args.max_iterations,
        )
        return result
    except Exception as e:
        logger.error(f"GPTFUZZER run failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"GPTFUZZER failed: {str(e)}"
        ) from e


# =============================================================================
# Analytics Endpoints
# =============================================================================


@app.post(
    "/jailbreakhub-analytics",
    response_model=JailbreakHubResponse,
    tags=["Analytics"],
    responses={
        **COMMON_RESPONSES,
        200: {"description": "JAILBREAKHUB analytics complete"},
    },
)
@limiter.limit(RATE_LIMIT_BATCH)
def jailbreakhub_analytics(
    request: Request,
    args: JailbreakHubRequest,
    api_key: APIKeyDep,
):
    """Cluster jailbreak prompts and return simple analytics."""
    logger.info(
        f"JAILBREAKHUB analytics: prompts={len(args.prompts)}, "
        f"max_samples={args.max_samples}"
    )
    try:
        return jailbreakhub_service(args)
    except Exception as e:
        logger.error(f"JAILBREAKHUB analytics failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"JAILBREAKHUB failed: {str(e)}"
        ) from e
