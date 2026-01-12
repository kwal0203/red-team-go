"""Curiosity-Driven Red Teaming (CRT) service adapter (placeholder).

Prototype path: experimental/iterative/curiosity-driven/main.py
"""

from typing import Any


def crt_service(
    model: dict[str, Any],
    prompt: str,
    max_steps: int = 100,
) -> dict[str, Any]:
    """Run curiosity-driven PPO-style exploration (placeholder)."""
    raise NotImplementedError(
        "CRT prototype not yet integrated. Run "
        "experimental/iterative/curiosity-driven/main.py and wrap outputs into a service adapter."
    )
