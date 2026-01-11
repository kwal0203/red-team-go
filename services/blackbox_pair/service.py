"""Black-box PAIR service adapter (placeholder).

Prototype path: experimental/iterative/black-box/main.py
"""

from typing import Any


def blackbox_pair_service(
    model: dict[str, Any],
    prompt: str,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Run black-box PAIR attacker/target/judge refinement (placeholder)."""
    raise NotImplementedError(
        "Black-box PAIR prototype not yet integrated. Run "
        "experimental/iterative/black-box/main.py and wrap outputs into a service adapter."
    )
