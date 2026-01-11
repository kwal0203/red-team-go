"""Don't-Say-No (DSN) service adapter (placeholder).

Prototype path: experimental/single/dont-say-no/main.py
"""

from typing import Any


def dsn_service(
    model: dict[str, Any],
    prompt: str,
    max_iterations: int = 50,
) -> dict[str, Any]:
    """Run DSN gradient-based suffix optimization (placeholder)."""
    raise NotImplementedError(
        "DSN prototype not yet integrated. Run "
        "experimental/single/dont-say-no/main.py and wrap outputs into a "
        "service adapter."
    )
