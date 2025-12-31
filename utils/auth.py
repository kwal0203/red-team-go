"""API Key authentication utilities."""

import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from utils.config import get_api_keys

logger = logging.getLogger(__name__)

# Define the header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_valid_api_keys() -> set[str]:
    """
    Load valid API keys from environment.

    Returns a set of valid API keys for O(1) lookup.
    """
    keys_str = get_api_keys()
    if not keys_str:
        return set()
    return {key.strip() for key in keys_str.split(",") if key.strip()}


async def verify_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> str:
    """
    Verify the API key from the X-API-Key header.

    This is a FastAPI dependency that can be injected into protected routes.

    Args:
        request: The incoming request (for logging)
        api_key: The API key from the header

    Returns:
        The validated API key

    Raises:
        HTTPException: 401 if key is missing or invalid
    """
    valid_keys = get_valid_api_keys()

    # If no API keys configured, allow access (dev mode warning)
    if not valid_keys:
        logger.warning(
            "No API keys configured - authentication disabled. "
            "Set REDTEAM_API_KEYS environment variable in production."
        )
        return "dev-mode"

    client_host = request.client.host if request.client else "unknown"

    if api_key is None:
        logger.warning(f"Missing API key from {client_host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    key_valid = any(
        secrets.compare_digest(api_key, valid_key) for valid_key in valid_keys
    )

    if not key_valid:
        logger.warning(f"Invalid API key attempt from {client_host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


# Type alias for dependency injection
APIKeyDep = Annotated[str, Depends(verify_api_key)]
