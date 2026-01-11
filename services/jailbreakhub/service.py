"""JAILBREAKHUB clustering/ASR service adapter (lightweight implementation).

This performs simple clustering (by prompt length buckets) and returns clusters.
It does not call external models; ASR replay can be added later.
"""

import uuid

from services.jailbreakhub.api import (
    JailbreakHubCluster,
    JailbreakHubRequest,
    JailbreakHubResponse,
)


def jailbreakhub_service(request: JailbreakHubRequest) -> JailbreakHubResponse:
    """Cluster jailbreak prompts using a simple length-based heuristic."""
    prompts = request.prompts[: request.max_samples]
    buckets: dict[int, list[str]] = {0: [], 1: [], 2: []}

    for p in prompts:
        length = len(p)
        if length < 80:
            buckets[0].append(p)
        elif length < 200:
            buckets[1].append(p)
        else:
            buckets[2].append(p)

    clusters = [
        JailbreakHubCluster(cluster_id=i, members=members)
        for i, members in buckets.items()
        if members
    ]

    metadata = {
        "method": "jailbreakhub",
        "run_id": f"jbhub-{uuid.uuid4().hex[:8]}",
        "prototype_path": "experimental/iterative/do-anything/main.py",
    }

    return JailbreakHubResponse(
        method="jailbreakhub",
        total_prompts=len(prompts),
        clusters=clusters,
        metadata=metadata,
    )
