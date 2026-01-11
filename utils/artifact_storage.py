"""Artifact storage utilities for evaluation results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ArtifactStorageError(RuntimeError):
    """Raised when artifact storage fails."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _scrub_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            lower_key = str(key).lower()
            if any(
                token in lower_key for token in ("key", "token", "secret", "password")
            ):
                redacted[key] = "***redacted***"
            else:
                redacted[key] = _scrub_secrets(value)
        return redacted
    if isinstance(payload, list):
        return [_scrub_secrets(item) for item in payload]
    return payload


def _to_serializable(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if hasattr(payload, "to_dict"):
        try:
            return payload.to_dict()
        except Exception:
            pass
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, set):
        return list(payload)
    return payload


def _build_key(
    prefix: str, evaluation_type: str, created_at: str, artifact_id: str
) -> str:
    date_prefix = created_at.split("T", maxsplit=1)[0]
    safe_prefix = prefix.strip("/") if prefix else "redteamgo"
    safe_eval = evaluation_type.strip("/")
    return f"{safe_prefix}/{safe_eval}/{date_prefix}/{artifact_id}.json"


class ArtifactStore:
    """Base artifact storage interface."""

    def save(self, key: str, data: bytes, content_type: str) -> None:
        raise NotImplementedError


class NullArtifactStore(ArtifactStore):
    """No-op storage."""

    def save(self, key: str, data: bytes, content_type: str) -> None:
        return


class LocalArtifactStore(ArtifactStore):
    """Local filesystem storage for artifacts."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    def save(self, key: str, data: bytes, content_type: str) -> None:
        destination = self.base_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


class S3ArtifactStore(ArtifactStore):
    """S3-backed storage for artifacts."""

    def __init__(self, bucket: str, region: str | None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - guarded by runtime config
            raise ArtifactStorageError(
                "boto3 is required for S3 artifact storage"
            ) from exc

        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)

    def save(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )


def _get_storage_mode() -> str:
    return os.getenv("ARTIFACT_STORAGE_MODE", "disabled").lower()


def _get_store() -> ArtifactStore:
    mode = _get_storage_mode()
    if mode in {"disabled", "none", "off"}:
        return NullArtifactStore()
    if mode == "local":
        base_dir = os.getenv("ARTIFACT_LOCAL_DIR", "artifacts")
        return LocalArtifactStore(base_dir)
    if mode == "s3":
        bucket = os.getenv("ARTIFACT_S3_BUCKET")
        if not bucket:
            raise ArtifactStorageError("ARTIFACT_S3_BUCKET is required for S3 mode")
        region = os.getenv("ARTIFACT_AWS_REGION")
        return S3ArtifactStore(bucket=bucket, region=region)
    raise ArtifactStorageError(f"Unknown ARTIFACT_STORAGE_MODE '{mode}'")


def store_evaluation_artifact(
    request: Request,
    evaluation_type: str,
    request_payload: Any,
    response_payload: Any,
    api_key: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Persist an evaluation artifact with provenance metadata."""

    try:
        store = _get_store()
    except ArtifactStorageError as exc:
        logger.error(f"Artifact storage unavailable: {exc}")
        return None

    if isinstance(store, NullArtifactStore):
        return None

    created_at = _now_iso()
    artifact_id = uuid.uuid4().hex
    key_prefix = os.getenv("ARTIFACT_S3_PREFIX", "redteamgo")
    key = _build_key(key_prefix, evaluation_type, created_at, artifact_id)

    metadata = {
        "artifact_id": artifact_id,
        "created_at": created_at,
        "evaluation_type": evaluation_type,
        "endpoint": request.url.path,
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "api_key_hash": _hash_api_key(api_key),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    request_serialized = _scrub_secrets(_to_serializable(request_payload))
    response_serialized = _scrub_secrets(_to_serializable(response_payload))

    payload = {
        "metadata": metadata,
        "request": request_serialized,
        "response": response_serialized,
    }

    data = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")

    try:
        store.save(key, data, content_type="application/json")
        logger.info(
            "Stored evaluation artifact",
            extra={"artifact_key": key, "evaluation_type": evaluation_type},
        )
        return key
    except Exception as exc:
        logger.error(f"Failed to store artifact: {exc}")
        return None
