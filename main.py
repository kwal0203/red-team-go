"""RedTeamGo - LLM Red Teaming and Safety Evaluation API."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from services.bias_detection_dbias.service import (
    bias_detection_realtime_service,
    dbias_service,
)
from services.guardrails.service import (
    guardrails_evaluate_service,
    guardrails_protect_service,
)
from services.toxicity_detection.service import (
    toxicity_detection_realtime_service,
    toxicity_detection_service,
)
from utils.models import (
    DetectionBatchBias,
    DetectionBatchToxicity,
    DetectionRealtimeBias,
    DetectionRealtimeToxicity,
    GuardrailEvaluateRequest,
    GuardrailEvaluateResponse,
    GuardrailProtectRequest,
    GuardrailProtectResponse,
    ResultBatch,
    ResultRealtimeBias,
    ResultRealtimeToxicity,
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
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy"}


@app.get("/")
def read_root():
    """Root endpoint returning service status."""
    return {"service": "online"}


# =============================================================================
# Batch Detection Endpoints
# =============================================================================


@app.post("/toxicity-detection-batch", response_model=ResultBatch)
def toxicity_detection_batch(args: DetectionBatchToxicity):
    """
    Batch toxicity detection endpoint.

    Evaluates multiple prompts for toxicity using an ensemble of moderator models
    (OpenAI Moderator + Paradetox).

    Args:
        args: Batch detection configuration including model, num_samples, and prompt source.

    Returns:
        ResultBatch containing toxicity evaluation results for all samples.
    """
    logger.info(f"Toxicity batch detection request: {args.num_samples} samples")
    toxicity_result = toxicity_detection_service(**args.model_dump())
    return ResultBatch(result=toxicity_result)


@app.post("/bias-detection-batch", response_model=ResultBatch)
def bias_detection_batch(args: DetectionBatchBias):
    """
    Batch bias detection endpoint.

    Evaluates multiple prompts for bias using self-evaluation methodology.

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
def toxicity_detection_realtime(args: DetectionRealtimeToxicity):
    """
    Realtime toxicity detection endpoint.

    Sends a single prompt to the target model, evaluates the response for toxicity,
    and returns both the response and toxicity scores.

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
def bias_detection_realtime(args: DetectionRealtimeBias):
    """
    Realtime bias detection endpoint.

    Sends a single prompt to the target model, evaluates the response for bias,
    and returns both the response and bias assessment.

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
def evaluate_guardrails(args: GuardrailEvaluateRequest):
    """
    Evaluate guardrails for red-team testing.

    Sends a prompt to the target model and evaluates both the input prompt
    and the model's response for various safety violations. Use this to test
    whether a model's guardrails can be bypassed.

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
def protect_guardrails(args: GuardrailProtectRequest):
    """
    Protect mode guardrail check for production middleware.

    Checks input and/or output text for safety violations and applies
    the specified remediation action (block, flag, or redact).

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
