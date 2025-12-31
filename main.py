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
from services.guardrails.service import (
    guardrails_evaluate_service,
    guardrails_protect_service,
)
from services.prompt_generation.service import prompt_generation_service
from services.stereotype_benchmarks.service import stereotype_benchmark_service
from services.toxicity_detection.service import (
    toxicity_detection_realtime_service,
    toxicity_detection_service,
)
from utils.auth import APIKeyDep
from utils.models import (
    AdversarialRobustnessRequest,
    AdversarialRobustnessResponse,
    DetectionBatchBias,
    DetectionBatchToxicity,
    DetectionRealtimeBias,
    DetectionRealtimeToxicity,
    GuardrailEvaluateRequest,
    GuardrailEvaluateResponse,
    GuardrailProtectRequest,
    GuardrailProtectResponse,
    PromptGenerationRequest,
    PromptGenerationResponse,
    ResultBatch,
    ResultRealtimeBias,
    ResultRealtimeToxicity,
    StereotypeBenchmarkRequest,
    StereotypeBenchmarkResponse,
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

# Create FastAPI app
app = FastAPI(
    title="RedTeamGo",
    description="LLM Red Teaming and Safety Evaluation API",
    version="0.1.0",
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


@app.get("/health")
@limiter.limit(RATE_LIMIT_HEALTH)
async def health_check(request: Request):
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/")
@limiter.limit(RATE_LIMIT_HEALTH)
def read_root(request: Request):
    """Root endpoint returning service status."""
    return {"service": "online"}


# =============================================================================
# Batch Detection Endpoints
# =============================================================================


@app.post("/toxicity-detection-batch", response_model=ResultBatch)
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
        args: Batch detection configuration including model, num_samples, and prompt source.

    Returns:
        ResultBatch containing toxicity evaluation results for all samples.
    """
    logger.info(f"Toxicity batch detection request: {args.num_samples} samples")
    toxicity_result = toxicity_detection_service(**args.model_dump())
    return ResultBatch(result=toxicity_result)


@app.post("/bias-detection-batch", response_model=ResultBatch)
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
        args: Batch detection configuration including model, num_samples, and prompt source.

    Returns:
        ResultBatch containing bias evaluation results for all samples.
    """
    logger.info(f"Bias batch detection request: {args.num_samples} samples")
    dbias_result = dbias_service(**args.model_dump())
    return ResultBatch(result=dbias_result)


# =============================================================================
# Realtime Detection Endpoints
# =============================================================================


@app.post("/toxicity-detection-realtime", response_model=ResultRealtimeToxicity)
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
        return result
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Toxicity detection failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Detection failed: {str(e)}"
        ) from e


@app.post("/bias-detection-realtime", response_model=ResultRealtimeBias)
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
        return result
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


@app.post("/evaluate/guardrails", response_model=GuardrailEvaluateResponse)
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
        return GuardrailEvaluateResponse(**result)
    except ValueError as e:
        logger.error(f"Invalid model configuration: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Guardrail evaluation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Evaluation failed: {str(e)}"
        ) from e


@app.post("/protect/guardrails", response_model=GuardrailProtectResponse)
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
        return GuardrailProtectResponse(**result)
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


@app.post("/adversarial-robustness", response_model=AdversarialRobustnessResponse)
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
        return AdversarialRobustnessResponse(**result)
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


@app.post("/stereotype-benchmark", response_model=StereotypeBenchmarkResponse)
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
        return StereotypeBenchmarkResponse(**result)
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


@app.post("/generate-adversarial-prompts", response_model=PromptGenerationResponse)
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
    - "llm": LLM-based prompt generation (implemented)
    - "genetic": Genetic algorithm (stub)
    - "pair": PAIR method (stub)

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
        return PromptGenerationResponse(**result)
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
