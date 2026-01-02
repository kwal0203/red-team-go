import json
from pathlib import Path

from starlette.requests import Request

from utils.artifact_storage import store_evaluation_artifact


def build_request(path: str = "/test") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"user-agent", b"pytest")],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def test_store_artifact_local(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_STORAGE_MODE", "local")
    monkeypatch.setenv("ARTIFACT_LOCAL_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_S3_PREFIX", "redteamgo")

    request = build_request("/toxicity-detection-batch")

    request_payload = {
        "model": {"name": "openai-gpt-4"},
        "prompt": "test",
        "api_key": "should-not-store",
    }
    response_payload = {"result": {"score": 0.1}}

    key = store_evaluation_artifact(
        request,
        "toxicity_batch",
        request_payload,
        response_payload,
        api_key="secret-key",
        extra_metadata={"model": "openai-gpt-4"},
    )

    assert key is not None
    artifact_path = Path(tmp_path) / key
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text())
    assert payload["metadata"]["evaluation_type"] == "toxicity_batch"
    assert payload["metadata"]["api_key_hash"] is not None
    assert payload["request"]["api_key"] == "***redacted***"
