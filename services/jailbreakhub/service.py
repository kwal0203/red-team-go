"""JAILBREAKHUB clustering/ASR service adapter (lightweight implementation).

This performs simple clustering (by prompt length buckets) and returns clusters.
It does not call external models; ASR replay can be added later.
"""

import json
import os
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

    response = JailbreakHubResponse(
        method="jailbreakhub",
        total_prompts=len(prompts),
        clusters=clusters,
        metadata=metadata,
    )
    _persist_results("jailbreakhub", metadata["run_id"], response)
    return response


def _persist_results(method: str, run_id: str, response: JailbreakHubResponse) -> None:
    """Persist analytics results to results/{method}/{run_id}.json."""
    results_dir = os.path.join("results", method)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
