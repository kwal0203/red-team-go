"""AutoDAN service adapter (placeholder).

Prototype path: experimental/iterative/autodan/main.py
"""

from typing import Any


def autodan_service(
    model: dict[str, Any],
    prompt: str,
    population_size: int = 20,
    generations: int = 10,
) -> dict[str, Any]:
    """Run AutoDAN genetic jailbreak evolution (placeholder)."""
    raise NotImplementedError(
        "AutoDAN prototype not yet integrated. Run "
        "experimental/iterative/autodan/main.py and wrap outputs into a service adapter."
    )
