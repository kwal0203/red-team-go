"""Rate limiting configuration using slowapi."""

import logging

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limit constants
RATE_LIMIT_BATCH = "10/minute"  # Heavy endpoints: batch operations
RATE_LIMIT_REALTIME = "30/minute"  # Medium endpoints: realtime detection
RATE_LIMIT_HEALTH = "60/minute"  # Light endpoints: health checks


def get_api_key_or_ip(request: Request) -> str:
    """
    Get the rate limit key from API key or IP address.

    Uses API key if present for per-client limiting,
    falls back to IP address for unauthenticated requests.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Use first 8 chars of API key for privacy in logs
        return f"key:{api_key[:8]}..."
    return str(get_remote_address(request))


# Create the limiter with in-memory storage
limiter = Limiter(
    key_func=get_api_key_or_ip,
    default_limits=["100/minute"],  # Default fallback
    storage_uri="memory://",
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Returns a JSON response with proper 429 status and retry-after header.
    """
    logger.warning(
        f"Rate limit exceeded for {get_api_key_or_ip(request)}: {exc.detail}"
    )

    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down.",
            "error": "rate_limit_exceeded",
            "limit": str(exc.detail),
        },
        headers={
            "Retry-After": "60",
            "X-RateLimit-Limit": str(exc.detail),
        },
    )
