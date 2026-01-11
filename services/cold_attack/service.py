"""COLD-Attack service adapter (placeholder).

Prototype path: experimental/iterative/cold-attack/main.py
"""

from typing import Any


def cold_attack_service(
    model: dict[str, Any],
    prompt: str,
    steps: int = 50,
) -> dict[str, Any]:
    """Run COLD-Attack soft-prompt optimization (placeholder)."""
    raise NotImplementedError(
        "COLD-Attack prototype not yet integrated. Run "
        "experimental/iterative/cold-attack/main.py and wrap outputs into a service adapter."
    )
