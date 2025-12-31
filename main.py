"""RedTeamGo - LLM Red Teaming and Safety Evaluation API."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from services.bias_detection_dbias.service import (
    bias_detection_realtime_service,
    dbias_service,
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
